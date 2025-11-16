// Function: sub_2152590
// Address: 0x2152590
//
void __fastcall sub_2152590(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r8
  int v6; // eax
  __int64 v7; // r12
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rax
  int v10; // ebx
  char v11; // r15
  unsigned int v12; // eax
  unsigned int i; // r12d
  unsigned int v14; // esi
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r12
  __int64 v28; // rax
  unsigned int v29; // [rsp+8h] [rbp-68h]
  int v30; // [rsp+Ch] [rbp-64h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  int v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+18h] [rbp-58h]
  unsigned __int64 v37; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v38; // [rsp+28h] [rbp-48h]
  char *v39; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v40; // [rsp+38h] [rbp-38h]

  v5 = sub_396DDB0();
  v6 = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)v6 == 13 )
  {
    v38 = *(_DWORD *)(a2 + 32);
    if ( v38 > 0x40 )
    {
      v35 = v5;
      sub_16A4FD0((__int64)&v37, (const void **)(a2 + 24));
      v5 = v35;
    }
    else
    {
      v37 = *(_QWORD *)(a2 + 24);
    }
    v7 = *(_QWORD *)a2;
    v33 = v5;
    v8 = (unsigned int)sub_15A9FE0(v5, *(_QWORD *)a2);
    v9 = v8 * ((v8 + ((unsigned __int64)(sub_127FA20(v33, v7) + 7) >> 3) - 1) / v8);
    v10 = 0;
    v34 = v9;
    if ( (_DWORD)v9 )
    {
      do
      {
        while ( 1 )
        {
          sub_16A88B0((__int64)&v39, (__int64)&v37, 8u);
          v11 = (char)v39;
          if ( v40 > 0x40 )
          {
            v11 = *v39;
            j_j___libc_free_0_0(v39);
          }
          *(_BYTE *)(*(_QWORD *)(a3 + 8) + *(unsigned int *)(a3 + 160)) = v11;
          v12 = v38;
          ++*(_DWORD *)(a3 + 160);
          if ( v12 <= 0x40 )
            break;
          ++v10;
          sub_16A8110((__int64)&v37, 8u);
          if ( v34 == v10 )
            goto LABEL_13;
        }
        if ( v12 == 8 )
          v37 = 0;
        else
          v37 >>= 8;
        ++v10;
      }
      while ( v34 != v10 );
    }
LABEL_13:
    if ( v38 > 0x40 )
    {
      if ( v37 )
        j_j___libc_free_0_0(v37);
    }
  }
  else if ( (((_BYTE)v6 - 6) & 0xFD) != 0 )
  {
    if ( (unsigned int)(v6 - 11) > 1 )
    {
      v30 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      if ( v30 )
      {
        v16 = 0;
        v29 = (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) - 1;
        v36 = *(_QWORD *)a2;
        do
        {
          v32 = v5;
          if ( (_DWORD)v16 == v29 )
          {
            v27 = *(_QWORD *)(sub_15A9930(v5, v36) + 16);
            LODWORD(v27) = sub_12BE0A0(v32, v36) + v27;
            v28 = sub_15A9930(v32, v36);
            v21 = v32;
            v22 = (unsigned int)(v27 - *(_DWORD *)(v28 + 8LL * v29 + 16));
          }
          else
          {
            v19 = *(_QWORD *)(sub_15A9930(v5, v36) + 8LL * (unsigned int)(v16 + 1) + 16);
            v20 = sub_15A9930(v32, v36);
            v21 = v32;
            v22 = (unsigned int)(v19 - *(_DWORD *)(v20 + 8LL * (unsigned int)v16 + 16));
          }
          if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
            v17 = *(_QWORD *)(a2 - 8);
          else
            v17 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v31 = v21;
          v18 = *(_QWORD *)(v17 + 24 * v16++);
          sub_21528B0(a1, v18, v22, a3);
          v5 = v31;
        }
        while ( v30 != (_DWORD)v16 );
      }
    }
    else if ( (unsigned int)sub_15958F0(a2) )
    {
      for ( i = 0; (unsigned int)sub_15958F0(a2) > i; ++i )
      {
        v14 = i;
        v15 = sub_15A0940(a2, v14);
        sub_21528B0(a1, v15, 0, a3);
      }
    }
  }
  else if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
  {
    v23 = 0;
    v24 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    do
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v25 = *(_QWORD *)(a2 - 8);
      else
        v25 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v26 = *(_QWORD *)(v25 + v23);
      v23 += 24;
      sub_21528B0(a1, v26, 0, a3);
    }
    while ( v24 != v23 );
  }
}
