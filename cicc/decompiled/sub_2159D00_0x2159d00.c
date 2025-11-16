// Function: sub_2159D00
// Address: 0x2159d00
//
__int64 __fastcall sub_2159D00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  int v4; // eax
  __int64 v5; // r12
  __int64 v6; // rbx
  _QWORD *v7; // r15
  unsigned __int8 v8; // al
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // esi
  __int64 *v12; // rcx
  __int64 v13; // r9
  unsigned int v14; // esi
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdi
  unsigned int v20; // edx
  int v21; // ecx
  __int64 v22; // rsi
  int v23; // r10d
  __int64 *v24; // r9
  int v25; // ecx
  int v26; // r10d
  int v27; // r11d
  __int64 *v28; // r9
  int v29; // r9d
  unsigned int v30; // ebx
  __int64 *v31; // r8
  __int64 v32; // rdx
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+20h] [rbp-50h] BYREF
  __int64 v36; // [rsp+28h] [rbp-48h]
  __int64 v37; // [rsp+30h] [rbp-40h]
  unsigned int v38; // [rsp+38h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 32);
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v34 = a2 + 24;
  if ( v3 == a2 + 24 )
  {
    v18 = 0;
    return j___libc_free_0(v18);
  }
LABEL_8:
  while ( 2 )
  {
    v5 = v3 - 56;
    if ( !v3 )
      v5 = 0;
    if ( !sub_15E4F60(v5) )
    {
      v6 = *(_QWORD *)(v5 + 8);
      if ( !v6 )
      {
LABEL_23:
        v14 = v38;
        if ( !v38 )
          goto LABEL_30;
        goto LABEL_24;
      }
      while ( 1 )
      {
        v7 = sub_1648700(v6);
        v8 = *((_BYTE *)v7 + 16);
        if ( v8 <= 0x10u )
        {
          if ( (unsigned __int8)sub_214AC00((__int64)v7) || (unsigned __int8)sub_214BAF0((__int64)v7, (__int64)&v35) )
          {
LABEL_29:
            sub_2151D30(a1, v5, a3);
            v14 = v38;
            if ( !v38 )
            {
LABEL_30:
              ++v35;
              goto LABEL_31;
            }
LABEL_24:
            v15 = (v14 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v16 = (__int64 *)(v36 + 16LL * v15);
            v17 = *v16;
            if ( v5 != *v16 )
            {
              v27 = 1;
              v28 = 0;
              while ( v17 != -8 )
              {
                if ( !v28 && v17 == -16 )
                  v28 = v16;
                v15 = (v14 - 1) & (v27 + v15);
                v16 = (__int64 *)(v36 + 16LL * v15);
                v17 = *v16;
                if ( v5 == *v16 )
                  goto LABEL_25;
                ++v27;
              }
              if ( v28 )
                v16 = v28;
              ++v35;
              v21 = v37 + 1;
              if ( 4 * ((int)v37 + 1) >= 3 * v14 )
              {
LABEL_31:
                sub_2159B40((__int64)&v35, 2 * v14);
                if ( v38 )
                {
                  v20 = (v38 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
                  v21 = v37 + 1;
                  v16 = (__int64 *)(v36 + 16LL * v20);
                  v22 = *v16;
                  if ( v5 != *v16 )
                  {
                    v23 = 1;
                    v24 = 0;
                    while ( v22 != -8 )
                    {
                      if ( v22 == -16 && !v24 )
                        v24 = v16;
                      v20 = (v38 - 1) & (v23 + v20);
                      v16 = (__int64 *)(v36 + 16LL * v20);
                      v22 = *v16;
                      if ( v5 == *v16 )
                        goto LABEL_48;
                      ++v23;
                    }
                    if ( v24 )
                      v16 = v24;
                  }
                  goto LABEL_48;
                }
              }
              else
              {
                if ( v14 - HIDWORD(v37) - v21 > v14 >> 3 )
                {
LABEL_48:
                  LODWORD(v37) = v21;
                  if ( *v16 != -8 )
                    --HIDWORD(v37);
                  *v16 = v5;
                  *((_BYTE *)v16 + 8) = 0;
                  goto LABEL_25;
                }
                sub_2159B40((__int64)&v35, v14);
                if ( v38 )
                {
                  v29 = 1;
                  v30 = (v38 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
                  v31 = 0;
                  v21 = v37 + 1;
                  v16 = (__int64 *)(v36 + 16LL * v30);
                  v32 = *v16;
                  if ( v5 != *v16 )
                  {
                    while ( v32 != -8 )
                    {
                      if ( !v31 && v32 == -16 )
                        v31 = v16;
                      v30 = (v38 - 1) & (v29 + v30);
                      v16 = (__int64 *)(v36 + 16LL * v30);
                      v32 = *v16;
                      if ( v5 == *v16 )
                        goto LABEL_48;
                      ++v29;
                    }
                    if ( v31 )
                      v16 = v31;
                  }
                  goto LABEL_48;
                }
              }
              LODWORD(v37) = v37 + 1;
              BUG();
            }
LABEL_25:
            *((_BYTE *)v16 + 8) = 1;
            v3 = *(_QWORD *)(v3 + 8);
            if ( v34 == v3 )
              goto LABEL_26;
            goto LABEL_8;
          }
          v8 = *((_BYTE *)v7 + 16);
        }
        if ( v8 > 0x17u )
        {
          v9 = v7[5];
          if ( v9 )
          {
            v10 = *(_QWORD *)(v9 + 56);
            if ( v10 )
            {
              if ( v38 )
              {
                v11 = (v38 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
                v12 = (__int64 *)(v36 + 16LL * v11);
                v13 = *v12;
                if ( v10 == *v12 )
                {
LABEL_21:
                  if ( v12 != (__int64 *)(v36 + 16LL * v38) )
                    goto LABEL_29;
                }
                else
                {
                  v25 = 1;
                  while ( v13 != -8 )
                  {
                    v26 = v25 + 1;
                    v11 = (v38 - 1) & (v25 + v11);
                    v12 = (__int64 *)(v36 + 16LL * v11);
                    v13 = *v12;
                    if ( v10 == *v12 )
                      goto LABEL_21;
                    v25 = v26;
                  }
                }
              }
            }
          }
        }
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          goto LABEL_23;
      }
    }
    if ( *(_QWORD *)(v5 + 8) )
    {
      v4 = *(_DWORD *)(v5 + 36);
      if ( v4 )
      {
        if ( v4 == 3785 )
          *(_QWORD *)(a1 + 792) = v5;
      }
      else
      {
        sub_2151D30(a1, v5, a3);
      }
    }
    v3 = *(_QWORD *)(v3 + 8);
    if ( v34 != v3 )
      continue;
    break;
  }
LABEL_26:
  v18 = v36;
  return j___libc_free_0(v18);
}
