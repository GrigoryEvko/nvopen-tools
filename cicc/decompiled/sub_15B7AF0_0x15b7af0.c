// Function: sub_15B7AF0
// Address: 0x15b7af0
//
__int64 __fastcall sub_15B7AF0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r14d
  __int64 v6; // rax
  __int64 v8; // r12
  __int64 v9; // rcx
  int v10; // esi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // r8
  int v14; // edx
  int v15; // eax
  unsigned int v16; // edx
  _QWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rsi
  _QWORD *v20; // r13
  __int64 v21; // r8
  __int64 v22; // r14
  __int64 v23; // r8
  __int64 v24; // r10
  __int64 v25; // r11
  int v26; // [rsp+Ch] [rbp-94h]
  int v27; // [rsp+10h] [rbp-90h] BYREF
  __int64 v28; // [rsp+18h] [rbp-88h] BYREF
  __int64 v29; // [rsp+20h] [rbp-80h] BYREF
  int v30; // [rsp+28h] [rbp-78h] BYREF
  __int64 v31; // [rsp+30h] [rbp-70h] BYREF
  __int64 v32[3]; // [rsp+38h] [rbp-68h] BYREF
  int v33; // [rsp+50h] [rbp-50h]
  int v34; // [rsp+54h] [rbp-4Ch]
  int v36; // [rsp+5Ch] [rbp-44h] BYREF
  __int64 v37; // [rsp+60h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v8 = *(_QWORD *)(a1 + 8);
    v9 = *(unsigned int *)(*a2 + 8);
    v10 = *(unsigned __int16 *)(*a2 + 2);
    v27 = v10;
    v11 = *(_QWORD *)(v6 + 8 * (2 - v9));
    v12 = v6;
    v28 = v11;
    if ( *(_BYTE *)v6 != 15 )
      v12 = *(_QWORD *)(v6 - 8LL * *(unsigned int *)(v6 + 8));
    v29 = v12;
    v30 = *(_DWORD *)(v6 + 24);
    v13 = *(_QWORD *)(v6 + 8 * (1 - v9));
    v31 = v13;
    v32[0] = *(_QWORD *)(v6 + 8 * (3 - v9));
    v32[1] = *(_QWORD *)(v6 + 32);
    v32[2] = *(_QWORD *)(v6 + 40);
    v33 = *(_DWORD *)(v6 + 48);
    if ( *(_BYTE *)(v6 + 56) )
      v34 = *(_DWORD *)(v6 + 52);
    v36 = *(_DWORD *)(v6 + 28);
    v37 = *(_QWORD *)(v6 + 8 * (4LL - *(unsigned int *)(v6 + 8)));
    if ( v13 != 0
      && v11 != 0
      && v10 == 13
      && *(_BYTE *)v13 == 13
      && *(_QWORD *)(v13 + 8 * (7LL - *(unsigned int *)(v13 + 8))) )
    {
      v14 = sub_15B2D00(&v28, &v31);
    }
    else
    {
      v14 = sub_15B4C20(&v27, &v28, &v29, &v30, &v31, v32, &v36);
    }
    v15 = v4 - 1;
    v16 = (v4 - 1) & v14;
    v17 = (_QWORD *)(v8 + 8LL * v16);
    v18 = *a2;
    v19 = *v17;
    if ( *v17 == *a2 )
    {
LABEL_25:
      *a3 = v17;
      return 1;
    }
    else
    {
      v20 = 0;
      v26 = 1;
      while ( v19 != -8 )
      {
        if ( v19 == -16 )
        {
          if ( !v20 )
            v20 = v17;
        }
        else
        {
          v21 = *(unsigned int *)(v18 + 8);
          v22 = *(_QWORD *)(v18 + 8 * (2 - v21));
          if ( v22 )
          {
            v23 = *(_QWORD *)(v18 + 8 * (1 - v21));
            if ( *(_WORD *)(v18 + 2) == 13 )
            {
              if ( v23 )
              {
                if ( *(_BYTE *)v23 == 13 )
                {
                  if ( *(_QWORD *)(v23 + 8 * (7LL - *(unsigned int *)(v23 + 8))) )
                  {
                    if ( *(_WORD *)(v19 + 2) == 13 )
                    {
                      v24 = *(unsigned int *)(v19 + 8);
                      v25 = *(_QWORD *)(v19 + 8 * (2 - v24));
                      if ( v22 == v25 && v25 && v23 == *(_QWORD *)(v19 + 8 * (1 - v24)) )
                        goto LABEL_25;
                    }
                  }
                }
              }
            }
          }
        }
        v16 = v15 & (v26 + v16);
        v17 = (_QWORD *)(v8 + 8LL * v16);
        v19 = *v17;
        if ( *v17 == v18 )
          goto LABEL_25;
        ++v26;
      }
      if ( !v20 )
        v20 = v17;
      *a3 = v20;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
