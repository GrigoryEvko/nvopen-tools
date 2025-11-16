// Function: sub_2F32820
// Address: 0x2f32820
//
__int64 __fastcall sub_2F32820(__int64 a1, __int64 a2, int a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // r14
  unsigned __int64 v6; // rcx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 i; // r15
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  char *v23; // r8
  __int64 v24; // r9
  int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rsi
  char *v29; // [rsp+0h] [rbp-70h]
  _QWORD v30[3]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v32[8]; // [rsp+30h] [rbp-40h] BYREF

  v4 = sub_2EBEE10(a2, a3);
  v5 = *(_QWORD *)(v4 + 32);
  v6 = v4;
  v7 = *(_DWORD *)(v4 + 40) & 0xFFFFFF;
  if ( v7 )
  {
    v8 = *(_QWORD *)(v4 + 32);
    v9 = v5 + 40LL * (unsigned int)(v7 - 1) + 40;
    while ( *(_BYTE *)v8 || (*(_BYTE *)(v8 + 3) & 0x10) == 0 || a3 != *(_DWORD *)(v8 + 8) )
    {
      v8 += 40;
      if ( v9 == v8 )
        goto LABEL_7;
    }
    v5 = v8;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x400000000LL;
  }
  else
  {
LABEL_7:
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x400000000LL;
    if ( !v5 )
      return a1;
  }
  v10 = *(_QWORD *)(v6 + 24);
  v11 = *(_QWORD *)(v10 + 56);
  for ( i = v10 + 48; i != v11; v11 = *(_QWORD *)(v11 + 8) )
  {
    if ( (unsigned __int16)(*(_WORD *)(v11 + 68) - 14) <= 1u )
    {
      v13 = *(_QWORD *)(v11 + 32);
      if ( !*(_BYTE *)v13 )
      {
        v14 = *(unsigned int *)(v5 + 8);
        if ( *(_DWORD *)(v13 + 8) == (_DWORD)v14 )
        {
          v32[0] = 0;
          v16 = *(_QWORD *)(v11 + 32);
          if ( *(_BYTE *)(v16 + 40) == 1 )
            v17 = *(_QWORD *)(v16 + 64);
          else
            v17 = *(unsigned int *)(v16 + 48);
          v30[0] = v17;
          v18 = *(_QWORD *)(v16 + 104);
          v19 = *(_QWORD *)(v16 + 144);
          v30[1] = v18;
          v30[2] = v19;
          if ( v32 != (_QWORD *)(v11 + 56) )
          {
            v20 = *(_QWORD *)(v11 + 56);
            v32[0] = v20;
            if ( v20 )
              sub_B96E90((__int64)v32, v20, 1);
          }
          v21 = *(unsigned int *)(a1 + 8);
          v22 = *(_QWORD *)a1;
          v23 = (char *)v30;
          v24 = v21 + 1;
          v25 = *(_DWORD *)(a1 + 8);
          if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            if ( v22 > (unsigned __int64)v30 || (unsigned __int64)v30 >= v22 + 40 * v21 )
            {
              sub_2F32700(a1, v21 + 1, v21, v14, (__int64)v30, v24);
              v21 = *(unsigned int *)(a1 + 8);
              v22 = *(_QWORD *)a1;
              v23 = (char *)v30;
              v25 = *(_DWORD *)(a1 + 8);
            }
            else
            {
              v29 = (char *)v30 - v22;
              sub_2F32700(a1, v21 + 1, v21, v14, (__int64)v30, v24);
              v22 = *(_QWORD *)a1;
              v21 = *(unsigned int *)(a1 + 8);
              v23 = &v29[*(_QWORD *)a1];
              v25 = *(_DWORD *)(a1 + 8);
            }
          }
          v26 = v22 + 40 * v21;
          if ( v26 )
          {
            *(_QWORD *)v26 = *(_QWORD *)v23;
            *(_QWORD *)(v26 + 8) = *((_QWORD *)v23 + 1);
            *(_QWORD *)(v26 + 16) = *((_QWORD *)v23 + 2);
            *(_BYTE *)(v26 + 24) = v23[24];
            v27 = *((_QWORD *)v23 + 4);
            *(_QWORD *)(v26 + 32) = v27;
            if ( v27 )
              sub_B96E90(v26 + 32, v27, 1);
            v25 = *(_DWORD *)(a1 + 8);
          }
          v28 = v32[0];
          *(_DWORD *)(a1 + 8) = v25 + 1;
          if ( v28 )
            sub_B91220((__int64)v32, v28);
        }
      }
    }
  }
  return a1;
}
