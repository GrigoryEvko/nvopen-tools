// Function: sub_AFC300
// Address: 0xafc300
//
__int64 __fastcall sub_AFC300(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r15d
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned __int8 v9; // al
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int8 v13; // al
  __int64 *v14; // r14
  int v15; // eax
  __int64 v16; // rdi
  unsigned int v17; // eax
  int v18; // r9d
  _QWORD *v19; // r8
  _QWORD *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // [rsp+8h] [rbp-78h]
  int v23; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h]
  __int64 v27; // [rsp+38h] [rbp-48h]
  int v28; // [rsp+40h] [rbp-40h]
  int v29; // [rsp+44h] [rbp-3Ch] BYREF
  __int64 v30[7]; // [rsp+48h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = *a2;
    v24 = 0;
    v25 = 0;
    v22 = v6;
    v9 = *(_BYTE *)(v7 - 16);
    v10 = v7 - 16;
    if ( (v9 & 2) != 0 )
    {
      v11 = *(_QWORD *)(v7 - 32);
      v12 = v11 + 8LL * *(unsigned int *)(v7 - 24);
    }
    else
    {
      v11 = v10 - 8LL * ((v9 >> 2) & 0xF);
      v12 = v11 + 8LL * ((*(_WORD *)(v7 - 16) >> 6) & 0xF);
    }
    v26 = v11 + 8;
    v27 = (v12 - (v11 + 8)) >> 3;
    v28 = *(_DWORD *)(v7 + 4);
    v29 = (unsigned __int16)sub_AF2710(v7);
    v13 = *(_BYTE *)(v7 - 16);
    if ( (v13 & 2) != 0 )
      v14 = *(__int64 **)(v7 - 32);
    else
      v14 = (__int64 *)(v10 - 8LL * ((v13 >> 2) & 0xF));
    v30[0] = *v14;
    v23 = v28;
    v15 = sub_AFB9D0(&v23, &v29, v30);
    v16 = *a2;
    v17 = (v4 - 1) & v15;
    v18 = 1;
    v19 = 0;
    v20 = (_QWORD *)(v22 + 8LL * v17);
    v21 = *v20;
    if ( *a2 == *v20 )
    {
LABEL_16:
      *a3 = v20;
      return 1;
    }
    else
    {
      while ( v21 != -4096 )
      {
        if ( v21 != -8192 || v19 )
          v20 = v19;
        v17 = (v4 - 1) & (v18 + v17);
        v21 = *(_QWORD *)(v22 + 8LL * v17);
        if ( v21 == v16 )
        {
          v20 = (_QWORD *)(v22 + 8LL * v17);
          goto LABEL_16;
        }
        ++v18;
        v19 = v20;
        v20 = (_QWORD *)(v22 + 8LL * v17);
      }
      if ( !v19 )
        v19 = v20;
      *a3 = v19;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
