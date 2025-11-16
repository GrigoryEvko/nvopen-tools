// Function: sub_359C870
// Address: 0x359c870
//
void __fastcall sub_359C870(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v8; // rbx
  __int64 v9; // rsi
  int v10; // r9d
  char v11; // r8
  int v12; // edi
  int v13; // r12d
  unsigned int i; // eax
  unsigned int v15; // r13d
  __int64 v16; // rbx
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rbx
  unsigned __int64 v20; // rax
  int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rdx
  unsigned int v26; // eax
  unsigned int v27; // r13d
  unsigned int v28; // r12d
  int v29; // eax
  int v30; // r8d
  _QWORD *v31; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+18h] [rbp-88h]
  __int64 *v33; // [rsp+20h] [rbp-80h]
  int v34; // [rsp+2Ch] [rbp-74h]
  int v38; // [rsp+48h] [rbp-58h]
  int v39; // [rsp+4Ch] [rbp-54h]
  int v40; // [rsp+50h] [rbp-50h]
  unsigned int v41; // [rsp+54h] [rbp-4Ch]
  unsigned int v42; // [rsp+58h] [rbp-48h]
  unsigned int v44; // [rsp+64h] [rbp-3Ch] BYREF
  unsigned int *v45; // [rsp+68h] [rbp-38h] BYREF

  v5 = a1[6];
  v31 = a1 + 10;
  v32 = sub_2E311E0(v5);
  v33 = a1 + 11;
  if ( *(_QWORD *)(v5 + 56) != v32 )
  {
    v6 = *(_QWORD *)(v5 + 56);
    do
    {
      v8 = a1[6];
      v34 = 0;
      v9 = *(_QWORD *)(v6 + 32);
      v10 = *(_DWORD *)(v6 + 40) & 0xFFFFFF;
      if ( v10 == 1 )
      {
        v13 = 0;
      }
      else
      {
        v11 = 0;
        v12 = 0;
        v13 = 0;
        for ( i = 1; i != v10; i += 2 )
        {
          if ( v8 == *(_QWORD *)(v9 + 40LL * (i + 1) + 24) )
            v11 = 1;
          else
            v13 = *(_DWORD *)(v9 + 40LL * i + 8);
          if ( v8 == *(_QWORD *)(v9 + 40LL * (i + 1) + 24) )
            v12 = *(_DWORD *)(v9 + 40LL * i + 8);
        }
        if ( !v11 )
          v12 = 0;
        v34 = v12;
      }
      v15 = *(_DWORD *)(v9 + 8);
      v16 = *a1;
      v17 = sub_2EBEE10(a1[3], v15);
      v18 = sub_3598DB0(v16, v17);
      v19 = *a1;
      v41 = v18;
      v20 = sub_2EBEE10(a1[3], v34);
      v21 = sub_3598DB0(v19, v20);
      v44 = v15;
      v22 = (__int64)v33;
      v40 = v21;
      v23 = a1[12];
      if ( !v23 )
        goto LABEL_21;
      do
      {
        while ( 1 )
        {
          v24 = *(_QWORD *)(v23 + 16);
          v25 = *(_QWORD *)(v23 + 24);
          if ( v15 <= *(_DWORD *)(v23 + 32) )
            break;
          v23 = *(_QWORD *)(v23 + 24);
          if ( !v25 )
            goto LABEL_19;
        }
        v22 = v23;
        v23 = *(_QWORD *)(v23 + 16);
      }
      while ( v24 );
LABEL_19:
      if ( (__int64 *)v22 == v33 || v15 < *(_DWORD *)(v22 + 32) )
      {
LABEL_21:
        v45 = &v44;
        v22 = sub_359C130(v31, v22, &v45);
      }
      v38 = v13;
      v39 = v15;
      v26 = *(_DWORD *)(v22 + 36) - (*(_BYTE *)(v22 + 40) == 0);
      if ( a3 <= v26 )
        v26 = a3;
      v27 = 0;
      v42 = v26;
      do
      {
        v28 = a3 - v27;
        v29 = sub_359C720((__int64)a1, a3 - v27, v41, v34, v40, a4, a1[6]);
        v30 = v27;
        if ( !v29 )
          v29 = v38;
        ++v27;
        sub_3599870(a1, a2, a5, v28, v30, v6, v39, v29, 0);
      }
      while ( v42 >= v27 );
      if ( (*(_BYTE *)v6 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
          v6 = *(_QWORD *)(v6 + 8);
      }
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v32 != v6 );
  }
}
