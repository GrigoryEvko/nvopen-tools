// Function: sub_14CE140
// Address: 0x14ce140
//
void __fastcall sub_14CE140(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdi
  _QWORD *v9; // rcx
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // r12
  _BYTE *v16; // rbx
  __int64 v17; // rax
  _BYTE *v18; // rdx
  __int64 v19; // [rsp+0h] [rbp-A0h]
  __int64 v20; // [rsp+18h] [rbp-88h]
  __int64 v21; // [rsp+20h] [rbp-80h] BYREF
  _BYTE *v22; // [rsp+28h] [rbp-78h]
  _BYTE *v23; // [rsp+30h] [rbp-70h]
  __int64 v24; // [rsp+38h] [rbp-68h]
  int v25; // [rsp+40h] [rbp-60h]
  _BYTE v26[88]; // [rsp+48h] [rbp-58h] BYREF

  v1 = *(_DWORD *)(a1 + 176);
  v21 = 0;
  v22 = v26;
  v23 = v26;
  v24 = 4;
  v25 = 0;
  if ( v1 )
  {
    v20 = *(_QWORD *)(a1 + 168);
    v19 = v20 + 48LL * *(unsigned int *)(a1 + 184);
    if ( v20 != v19 )
    {
      while ( 1 )
      {
        v2 = *(_QWORD *)(v20 + 24);
        if ( v2 != -16 && v2 != -8 )
          break;
        v20 += 48;
        if ( v19 == v20 )
          return;
      }
      if ( v19 != v20 )
      {
        while ( 1 )
        {
          v3 = *(_QWORD *)(v20 + 40);
          if ( !*(_BYTE *)(v3 + 184) )
            sub_14CDF70(*(_QWORD *)(v20 + 40));
          v4 = *(_QWORD *)(v3 + 8);
          v5 = v4 + 32LL * *(unsigned int *)(v3 + 16);
          if ( v5 != v4 )
            break;
LABEL_25:
          v10 = *(_QWORD *)(v20 + 24);
          v11 = *(_QWORD *)(v10 + 80);
          if ( v11 != v10 + 72 )
          {
            while ( 1 )
            {
              if ( !v11 )
                BUG();
              v12 = *(_QWORD *)(v11 + 24);
              if ( v12 != v11 + 16 )
                break;
LABEL_51:
              v11 = *(_QWORD *)(v11 + 8);
              if ( v10 + 72 == v11 )
                goto LABEL_52;
            }
            while ( 1 )
            {
              if ( !v12 )
                BUG();
              if ( *(_BYTE *)(v12 - 8) != 78 )
                goto LABEL_29;
              v13 = *(_QWORD *)(v12 - 48);
              if ( *(_BYTE *)(v13 + 16) || *(_DWORD *)(v13 + 36) != 4 )
                goto LABEL_29;
              v14 = v22;
              v15 = v12 - 24;
              if ( v23 == v22 )
              {
                v16 = &v22[8 * HIDWORD(v24)];
                if ( v22 == v16 )
                {
                  v18 = v22;
                }
                else
                {
                  do
                  {
                    if ( v15 == *v14 )
                      break;
                    ++v14;
                  }
                  while ( v16 != (_BYTE *)v14 );
                  v18 = &v22[8 * HIDWORD(v24)];
                }
                goto LABEL_45;
              }
              v16 = &v23[8 * (unsigned int)v24];
              v14 = (_QWORD *)sub_16CC9F0(&v21, v12 - 24);
              if ( v15 == *v14 )
                break;
              if ( v23 == v22 )
              {
                v14 = &v23[8 * HIDWORD(v24)];
                v18 = v14;
                goto LABEL_45;
              }
              v14 = &v23[8 * (unsigned int)v24];
LABEL_38:
              if ( v16 == (_BYTE *)v14 )
                sub_16BD130("Assumption in scanned function not in cache", 1);
LABEL_29:
              v12 = *(_QWORD *)(v12 + 8);
              if ( v11 + 16 == v12 )
                goto LABEL_51;
            }
            if ( v23 == v22 )
              v18 = &v23[8 * HIDWORD(v24)];
            else
              v18 = &v23[8 * (unsigned int)v24];
LABEL_45:
            while ( v18 != (_BYTE *)v14 )
            {
              if ( *v14 < 0xFFFFFFFFFFFFFFFELL )
                break;
              ++v14;
            }
            goto LABEL_38;
          }
LABEL_52:
          v20 += 48;
          if ( v20 != v19 )
          {
            while ( 1 )
            {
              v17 = *(_QWORD *)(v20 + 24);
              if ( v17 != -16 && v17 != -8 )
                break;
              v20 += 48;
              if ( v19 == v20 )
                goto LABEL_56;
            }
            if ( v19 != v20 )
              continue;
          }
LABEL_56:
          if ( v23 != v22 )
            _libc_free((unsigned __int64)v23);
          return;
        }
        while ( 1 )
        {
LABEL_15:
          v6 = *(_QWORD *)(v4 + 16);
          if ( v6 )
          {
            v7 = v22;
            if ( v23 != v22 )
              goto LABEL_13;
            v8 = &v22[8 * HIDWORD(v24)];
            if ( v22 != (_BYTE *)v8 )
            {
              v9 = 0;
              while ( v6 != *v7 )
              {
                if ( *v7 == -2 )
                  v9 = v7;
                if ( v8 == ++v7 )
                {
                  if ( !v9 )
                    goto LABEL_61;
                  v4 += 32;
                  *v9 = v6;
                  --v25;
                  ++v21;
                  if ( v5 != v4 )
                    goto LABEL_15;
                  goto LABEL_25;
                }
              }
              goto LABEL_14;
            }
LABEL_61:
            if ( HIDWORD(v24) < (unsigned int)v24 )
            {
              ++HIDWORD(v24);
              *v8 = v6;
              ++v21;
            }
            else
            {
LABEL_13:
              sub_16CCBA0(&v21, v6);
            }
          }
LABEL_14:
          v4 += 32;
          if ( v5 == v4 )
            goto LABEL_25;
        }
      }
    }
  }
}
