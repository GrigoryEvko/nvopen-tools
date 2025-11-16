// Function: sub_324EE90
// Address: 0x324ee90
//
__int64 __fastcall sub_324EE90(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int8 v5; // al
  char *v6; // r12
  bool v7; // r14
  __int64 v8; // r12
  char v9; // cl
  __int64 result; // rax
  unsigned __int8 v11; // dl
  __int64 v12; // r10
  __int64 v13; // rax
  _BYTE **v14; // r13
  const void *v15; // r14
  size_t v16; // rdx
  size_t v17; // r10
  _BYTE *v18; // r15
  __int64 v19; // r12
  unsigned __int8 v20; // al
  char *v21; // [rsp+8h] [rbp-68h]
  char v22; // [rsp+10h] [rbp-60h]
  char v23; // [rsp+17h] [rbp-59h]
  size_t v24; // [rsp+18h] [rbp-58h]
  _BYTE **v25; // [rsp+28h] [rbp-48h]

  v3 = a3 - 16;
  v5 = *(_BYTE *)(a3 - 16);
  if ( (v5 & 2) == 0 )
  {
    v6 = *(char **)(v3 - 8LL * ((v5 >> 2) & 0xF) + 24);
    if ( v6 )
      goto LABEL_3;
LABEL_30:
    v7 = 0;
    if ( !*(_BYTE *)(a3 + 52) )
      goto LABEL_7;
    goto LABEL_31;
  }
  v6 = *(char **)(*(_QWORD *)(a3 - 32) + 24LL);
  if ( !v6 )
    goto LABEL_30;
LABEL_3:
  v7 = sub_32120E0(v6);
  if ( (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0 || (unsigned __int16)sub_3220AA0(a1[26]) > 2u )
    sub_32495E0(a1, a2, (__int64)v6, 73);
  if ( (unsigned __int16)sub_3220AA0(a1[26]) > 3u && (*(_BYTE *)(a3 + 23) & 1) != 0 )
    sub_3249FA0(a1, a2, 109);
  if ( *(_BYTE *)(a3 + 52) )
LABEL_31:
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 16369, 65547, *(unsigned int *)(a3 + 48));
LABEL_7:
  v23 = (*(_BYTE *)(a3 - 16) & 2) != 0;
  if ( (*(_BYTE *)(a3 - 16) & 2) != 0 )
  {
    v8 = *(_QWORD *)(a3 - 32);
    v21 = *(char **)(v8 + 8);
    if ( !v21 )
      goto LABEL_10;
    v23 = 0;
    v9 = *v21;
    if ( (unsigned __int8)*v21 > 0x21u )
      goto LABEL_10;
LABEL_28:
    v23 = (0x200230000uLL >> v9) & 1;
    goto LABEL_10;
  }
  v8 = v3 - 8LL * ((*(_BYTE *)(a3 - 16) >> 2) & 0xF);
  v21 = *(char **)(v8 + 8);
  if ( !v21 )
  {
    v23 = 1;
    goto LABEL_10;
  }
  v9 = *v21;
  if ( (unsigned __int8)*v21 <= 0x21u )
    goto LABEL_28;
LABEL_10:
  result = *(_QWORD *)(v8 + 32);
  if ( result )
  {
    v11 = *(_BYTE *)(result - 16);
    if ( (v11 & 2) != 0 )
    {
      v12 = *(_QWORD *)(result - 32);
      v13 = *(unsigned int *)(result - 24);
    }
    else
    {
      v12 = result - 16 - 8LL * ((v11 >> 2) & 0xF);
      v13 = (*(_WORD *)(result - 16) >> 6) & 0xF;
    }
    result = v12 + 8 * v13;
    v25 = (_BYTE **)result;
    if ( result != v12 )
    {
      result = v7;
      v14 = (_BYTE **)v12;
      v22 = v7;
      do
      {
        v18 = *v14;
        if ( !*v14 || *v18 != 11 )
          goto LABEL_19;
        v19 = sub_324C6D0(a1, 40, a2, 0);
        v20 = *(v18 - 16);
        if ( (v20 & 2) != 0 )
        {
          v15 = (const void *)**((_QWORD **)v18 - 4);
          if ( v15 )
            goto LABEL_16;
        }
        else
        {
          v15 = *(const void **)&v18[-8 * ((v20 >> 2) & 0xF) - 16];
          if ( v15 )
          {
LABEL_16:
            v15 = (const void *)sub_B91420((__int64)v15);
            v17 = v16;
            goto LABEL_17;
          }
        }
        v17 = 0;
LABEL_17:
        v24 = v17;
        sub_324AD70(a1, v19, 3, v15, v17);
        result = sub_324A2D0(a1, v19, (__int64)(v18 + 16), v22);
        if ( v23 )
          result = (*(__int64 (__fastcall **)(__int64 *, const void *, size_t, __int64, char *))(*a1 + 24))(
                     a1,
                     v15,
                     v24,
                     v19,
                     v21);
LABEL_19:
        ++v14;
      }
      while ( v25 != v14 );
    }
  }
  return result;
}
