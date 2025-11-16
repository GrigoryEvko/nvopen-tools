// Function: sub_BA1DF0
// Address: 0xba1df0
//
__int64 __fastcall sub_BA1DF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // r12d
  __int64 v4; // r15
  __int64 v5; // rdi
  __int64 v6; // rdx
  int v7; // ecx
  __int64 v8; // rbx
  unsigned int v9; // r15d
  __int64 *v10; // r14
  __int64 v11; // r12
  __int64 result; // rax
  unsigned int v13; // esi
  int v14; // eax
  __int64 *v15; // rdx
  int v16; // eax
  unsigned int v17; // edx
  __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-110h]
  __int64 *v20; // [rsp+10h] [rbp-100h]
  __int64 v21; // [rsp+18h] [rbp-F8h]
  int v22; // [rsp+30h] [rbp-E0h]
  int v23; // [rsp+34h] [rbp-DCh]
  __int64 v24; // [rsp+38h] [rbp-D8h] BYREF
  __int64 *v25; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v26; // [rsp+48h] [rbp-C8h]
  __int64 *v27; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v29; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v30; // [rsp+68h] [rbp-A8h] BYREF
  int v31; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v32[4]; // [rsp+78h] [rbp-98h] BYREF
  char v33; // [rsp+9Ch] [rbp-74h]
  __int64 v34; // [rsp+A8h] [rbp-68h]

  v2 = a2;
  v24 = a1;
  sub_B922F0((__int64)&v27, a1);
  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 8);
  if ( !v3 )
    goto LABEL_26;
  v25 = 0;
  v26 = 0;
  if ( v27 )
  {
    if ( *(_BYTE *)v27 == 14 )
    {
      v5 = sub_AF5140((__int64)v27, 7u);
      if ( v5 )
      {
        v25 = (__int64 *)sub_B91420(v5);
        v26 = v6;
      }
    }
  }
  v7 = (v33 & 8) == 0 && v29 && v27 && *(_BYTE *)v27 == 14
     ? sub_AFA7A0(&v29, &v25)
     : sub_AFA420(&v28, &v25, &v30, v32, &v31);
  v23 = 1;
  v22 = v3 - 1;
  v8 = v4;
  v9 = (v3 - 1) & v7;
  while ( 1 )
  {
    v10 = (__int64 *)(v8 + 8LL * v9);
    v11 = *v10;
    if ( *v10 == -4096 )
    {
      v2 = a2;
      goto LABEL_26;
    }
    if ( v11 != -8192 )
    {
      if ( (v33 & 8) == 0 && v27 != 0 && v29 != 0 )
      {
        v21 = v29;
        if ( *(_BYTE *)v27 == 14 )
        {
          v20 = v27;
          if ( sub_AF5140((__int64)v27, 7u) )
          {
            if ( (*(_BYTE *)(v11 + 36) & 8) == 0 )
            {
              v19 = v34;
              if ( v20 == *((__int64 **)sub_A17150((_BYTE *)(v11 - 16)) + 1) && v21 == sub_AF5140(v11, 3u) )
              {
                if ( (*(_BYTE *)(v11 - 16) & 2) != 0 )
                  v17 = *(_DWORD *)(v11 - 24);
                else
                  v17 = (*(_WORD *)(v11 - 16) >> 6) & 0xF;
                v18 = 0;
                if ( v17 > 9 )
                  v18 = *((_QWORD *)sub_A17150((_BYTE *)(v11 - 16)) + 9);
                if ( v19 == v18 )
                  break;
              }
            }
          }
        }
      }
      if ( sub_AF5710((__int64 *)&v27, v11) )
        break;
    }
    v9 = v22 & (v23 + v9);
    ++v23;
  }
  v2 = a2;
  if ( v10 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) || (result = v11) == 0 )
  {
LABEL_26:
    if ( (unsigned __int8)sub_AFDBF0(v2, &v24, &v25) )
      return v24;
    v13 = *(_DWORD *)(v2 + 24);
    v14 = *(_DWORD *)(v2 + 16);
    v15 = v25;
    ++*(_QWORD *)v2;
    v16 = v14 + 1;
    v27 = v15;
    if ( 4 * v16 >= 3 * v13 )
    {
      v13 *= 2;
    }
    else if ( v13 - *(_DWORD *)(v2 + 20) - v16 > v13 >> 3 )
    {
LABEL_30:
      *(_DWORD *)(v2 + 16) = v16;
      if ( *v15 != -4096 )
        --*(_DWORD *)(v2 + 20);
      *v15 = v24;
      return v24;
    }
    sub_B07C50(v2, v13);
    sub_AFDBF0(v2, &v24, &v27);
    v15 = v27;
    v16 = *(_DWORD *)(v2 + 16) + 1;
    goto LABEL_30;
  }
  return result;
}
