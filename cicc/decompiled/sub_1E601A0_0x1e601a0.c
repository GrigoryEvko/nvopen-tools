// Function: sub_1E601A0
// Address: 0x1e601a0
//
__int64 __fastcall sub_1E601A0(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // r8d
  int v9; // r9d
  unsigned __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rcx
  int v16; // r8d
  unsigned int v17; // r9d
  __int64 *v18; // r15
  __int64 v19; // rsi
  __int64 *v20; // rax
  __int64 *v21; // rdi
  __int64 *v22; // [rsp+18h] [rbp-2B8h]
  __int64 *v23; // [rsp+28h] [rbp-2A8h]
  unsigned int v24; // [rsp+34h] [rbp-29Ch]
  __int64 v25; // [rsp+38h] [rbp-298h] BYREF
  __int64 v26; // [rsp+40h] [rbp-290h] BYREF
  __int64 v27; // [rsp+48h] [rbp-288h] BYREF
  __int64 v28; // [rsp+50h] [rbp-280h] BYREF
  __int64 v29; // [rsp+58h] [rbp-278h] BYREF
  _BYTE *v30; // [rsp+60h] [rbp-270h] BYREF
  __int64 v31; // [rsp+68h] [rbp-268h]
  _BYTE v32[256]; // [rsp+70h] [rbp-260h] BYREF
  __int64 v33; // [rsp+170h] [rbp-160h] BYREF
  __int64 *v34; // [rsp+178h] [rbp-158h]
  __int64 *v35; // [rsp+180h] [rbp-150h]
  __int64 v36; // [rsp+188h] [rbp-148h]
  int v37; // [rsp+190h] [rbp-140h]
  _BYTE v38[312]; // [rsp+198h] [rbp-138h] BYREF

  v5 = (__int64)(a1 + 3);
  v25 = a2;
  v22 = sub_1E60050((__int64)(a1 + 3), &v25);
  if ( *((_DWORD *)v22 + 2) < a3 )
    return v25;
  v33 = 0;
  v30 = v32;
  v31 = 0x2000000000LL;
  v34 = (__int64 *)v38;
  v35 = (__int64 *)v38;
  v36 = 32;
  v37 = 0;
  if ( *((_DWORD *)v22 + 3) >= a3 )
  {
    sub_1E05890((__int64)&v30, &v25, v6, v7, v8, v9);
    v13 = (unsigned int)v31;
    if ( !(_DWORD)v31 )
    {
LABEL_23:
      v11 = v22[3];
      if ( v35 != v34 )
        _libc_free((unsigned __int64)v35);
      v10 = (unsigned __int64)v30;
      if ( v30 != v32 )
        goto LABEL_4;
      return v11;
    }
    while ( 1 )
    {
      v26 = *(_QWORD *)&v30[8 * v13 - 8];
      v18 = sub_1E60050(v5, &v26);
      v19 = *(_QWORD *)(*a1 + 8LL * *((unsigned int *)v18 + 3));
      v17 = *((_DWORD *)v18 + 3);
      v20 = v34;
      v27 = v19;
      if ( v35 == v34 )
      {
        v14 = HIDWORD(v36);
        v21 = &v34[HIDWORD(v36)];
        if ( v34 != v21 )
        {
          v15 = 0;
          while ( 1 )
          {
            v14 = *v20;
            if ( v19 == *v20 )
              goto LABEL_19;
            if ( v14 == -2 )
              v15 = v20;
            if ( v21 == ++v20 )
            {
              if ( !v15 )
                break;
              *v15 = v19;
              --v37;
              v17 = *((_DWORD *)v18 + 3);
              ++v33;
              if ( a3 > v17 )
                goto LABEL_10;
              goto LABEL_29;
            }
          }
        }
        if ( HIDWORD(v36) < (unsigned int)v36 )
        {
          ++HIDWORD(v36);
          *v21 = v19;
          v17 = *((_DWORD *)v18 + 3);
          ++v33;
LABEL_9:
          if ( a3 <= v17 )
          {
LABEL_29:
            sub_1E05890((__int64)&v30, &v27, v14, (__int64)v15, v16, v17);
            v13 = (unsigned int)v31;
          }
          else
          {
LABEL_10:
            v13 = (unsigned int)(v31 - 1);
            LODWORD(v31) = v31 - 1;
          }
          goto LABEL_11;
        }
      }
      sub_16CCBA0((__int64)&v33, v19);
      v17 = *((_DWORD *)v18 + 3);
      if ( (_BYTE)v14 )
        goto LABEL_9;
LABEL_19:
      v13 = (unsigned int)(v31 - 1);
      LODWORD(v31) = v31 - 1;
      if ( a3 > v17 )
      {
LABEL_11:
        if ( !(_DWORD)v13 )
          goto LABEL_23;
      }
      else
      {
        v23 = sub_1E60050(v5, &v27);
        v28 = v23[3];
        v29 = v18[3];
        v24 = *((_DWORD *)sub_1E60050(v5, &v28) + 4);
        if ( v24 < *((_DWORD *)sub_1E60050(v5, &v29) + 4) )
          v18[3] = v28;
        *((_DWORD *)v18 + 3) = *((_DWORD *)v23 + 3);
        v13 = (unsigned int)v31;
        if ( !(_DWORD)v31 )
          goto LABEL_23;
      }
    }
  }
  v10 = (unsigned __int64)v30;
  v11 = v22[3];
  if ( v30 != v32 )
LABEL_4:
    _libc_free(v10);
  return v11;
}
