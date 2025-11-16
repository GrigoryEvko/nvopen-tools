// Function: sub_8B8140
// Address: 0x8b8140
//
__int64 __fastcall sub_8B8140(
        _BYTE *a1,
        __int64 a2,
        __int64 a3,
        int a4,
        int a5,
        int a6,
        int a7,
        unsigned __int8 a8,
        _DWORD *a9)
{
  __m128i *v11; // r15
  __int64 v12; // rsi
  char v13; // al
  char v14; // al
  int v15; // ebx
  int v16; // edx
  unsigned __int64 v17; // r12
  __int64 v18; // rdi
  _QWORD **v19; // rdi
  unsigned int v21; // esi
  _BOOL4 v22; // eax
  __int64 *v23; // rax
  __int64 v25; // [rsp+8h] [rbp-88h]
  int v27; // [rsp+20h] [rbp-70h]
  int v28; // [rsp+24h] [rbp-6Ch]
  bool v31; // [rsp+37h] [rbp-59h]
  _BYTE *v32; // [rsp+38h] [rbp-58h] BYREF
  int v33; // [rsp+44h] [rbp-4Ch] BYREF
  __int64 v34; // [rsp+48h] [rbp-48h] BYREF
  _QWORD *v35; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v36[7]; // [rsp+58h] [rbp-38h] BYREF

  v11 = *(__m128i **)(a2 + 288);
  v32 = a1;
  v12 = (__int64)a1;
  *a9 = 0;
  v13 = a1[82];
  v34 = 0;
  v35 = 0;
  v31 = (v13 & 8) != 0;
  if ( (a1[81] & 0x10) != 0
    && (*(_BYTE *)(a3 + 18) & 1) == 0
    && (v12 = (__int64)v32, (v25 = sub_5EDAE0((__int64)a1, a2, 0, &v34)) != 0) )
  {
    if ( !(a7 | a5) )
      goto LABEL_44;
    v27 = 1;
    v12 = (__int64)v32;
  }
  else
  {
    v25 = 0;
    v27 = 0;
  }
  v14 = *(_BYTE *)(v12 + 80);
  v15 = 0;
  if ( v14 != 17 )
    goto LABEL_5;
  v12 = *(_QWORD *)(v12 + 88);
  v32 = (_BYTE *)v12;
  if ( v12 )
  {
    v14 = *(_BYTE *)(v12 + 80);
    v15 = 1;
LABEL_5:
    v16 = 0;
    while ( 1 )
    {
      v17 = v12;
      if ( v14 == 16 )
      {
        v17 = **(_QWORD **)(v12 + 88);
        v14 = *(_BYTE *)(v17 + 80);
      }
      if ( v14 == 24 )
      {
        v17 = *(_QWORD *)(v17 + 88);
        v14 = *(_BYTE *)(v17 + 80);
      }
      if ( v14 == 20 )
      {
        if ( !v31 || (v28 = v16, v22 = sub_8808B0((__int64)a1, v12), v16 = v28, v22) )
        {
          if ( !v16 )
          {
            if ( a7 )
            {
              v18 = **(_QWORD **)(*(_QWORD *)(v17 + 88) + 32LL);
              if ( (!v18 || a7 != *(_DWORD *)(sub_892BC0(v18) + 4)) && (!dword_4F077BC || v27) )
              {
                v19 = (_QWORD **)v35;
                v16 = 0;
                if ( v35 )
                  goto LABEL_22;
LABEL_30:
                if ( !v27 )
                {
                  if ( !v16 )
                    goto LABEL_32;
LABEL_39:
                  v21 = 493;
                  if ( !a6 )
                    goto LABEL_34;
                  return v25;
                }
                goto LABEL_44;
              }
            }
          }
          if ( (unsigned int)sub_8B8060(v17, v11, *(_QWORD *)(a3 + 40), 1, (*(_BYTE *)(a2 + 130) & 0xC) != 0) )
            sub_8B5FF0(&v35, v17, 0);
          v16 = 1;
        }
      }
      if ( !v15 )
      {
        v32 = 0;
LABEL_21:
        v19 = (_QWORD **)v35;
        if ( v35 )
          goto LABEL_22;
        goto LABEL_30;
      }
      v12 = *((_QWORD *)v32 + 1);
      v32 = (_BYTE *)v12;
      if ( !v12 )
        goto LABEL_21;
      v14 = *(_BYTE *)(v12 + 80);
    }
  }
  v19 = (_QWORD **)v35;
  if ( v35 )
  {
LABEL_22:
    sub_893120(v19, 0, (__int64)&v32, v36, &v33, 0);
    if ( v33 )
    {
      if ( a6 )
        return v25;
      sub_6854E0(0x134u, (__int64)a1);
    }
    else
    {
      v23 = sub_8B7F20(
              (unsigned __int64)v32,
              v11,
              *(_QWORD *)(a3 + 40),
              *(_BYTE *)(a3 + 18) & 1,
              1,
              (*(_BYTE *)(a2 + 130) & 0xC) != 0,
              a4,
              a9);
      if ( !v23 )
        v23 = (__int64 *)v25;
      v25 = (__int64)v23;
    }
    goto LABEL_44;
  }
  if ( v27 )
  {
LABEL_44:
    if ( v34 && !a6 )
      sub_6853B0(a8, 0x10Au, (FILE *)(a3 + 8), (__int64)a1);
    return v25;
  }
LABEL_32:
  v21 = 147;
  if ( a1[80] == 17 )
    goto LABEL_39;
  if ( !a6 )
LABEL_34:
    sub_6853B0(a8, v21, (FILE *)(a3 + 8), (__int64)a1);
  return v25;
}
