// Function: sub_127CB50
// Address: 0x127cb50
//
__int64 __fastcall sub_127CB50(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r13
  char v5; // al
  __int64 i; // r12
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 *j; // r12
  unsigned __int64 v11; // rcx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _BOOL4 v14; // r8d
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-80h]
  unsigned __int64 v17; // [rsp+8h] [rbp-78h]
  _BOOL4 v18; // [rsp+8h] [rbp-78h]
  unsigned __int64 v19; // [rsp+18h] [rbp-68h] BYREF
  char v20[8]; // [rsp+20h] [rbp-60h] BYREF
  int v21; // [rsp+28h] [rbp-58h] BYREF
  __int64 v22; // [rsp+30h] [rbp-50h]
  int *v23; // [rsp+38h] [rbp-48h]
  int *v24; // [rsp+40h] [rbp-40h]
  __int64 v25; // [rsp+48h] [rbp-38h]

  sub_127CD40(a1, *(_QWORD *)(a2 + 72));
  v2 = *(_QWORD *)(a2 + 72);
  v21 = 0;
  v22 = 0;
  v3 = *(_QWORD *)(v2 + 72);
  v23 = &v21;
  v24 = &v21;
  v25 = 0;
  if ( v3 )
  {
    v4 = a1 + 120;
    while ( 1 )
    {
      v5 = *(_BYTE *)(v3 + 40);
      if ( v5 == 20 )
        break;
      if ( v5 == 15 )
      {
        for ( i = (__int64)v23; (int *)i != &v21; i = sub_220EF30(i) )
        {
          v19 = *(_QWORD *)(i + 32);
          sub_91CFF0(v4, &v19);
        }
        v7 = v22;
        while ( v7 )
        {
          sub_127AA60(*(_QWORD *)(v7 + 24));
          v8 = v7;
          v7 = *(_QWORD *)(v7 + 16);
          j_j___libc_free_0(v8, 40);
        }
        v22 = 0;
        v23 = &v21;
        v24 = &v21;
        v25 = 0;
        v3 = *(_QWORD *)(v3 + 16);
        if ( !v3 )
        {
LABEL_11:
          v3 = v22;
          return sub_127AA60(v3);
        }
      }
      else
      {
LABEL_3:
        v3 = *(_QWORD *)(v3 + 16);
        if ( !v3 )
          goto LABEL_11;
      }
    }
    for ( j = *(__int64 **)(v3 + 72); j; j = (__int64 *)*j )
    {
      if ( *((_BYTE *)j + 8) == 7 )
      {
        v11 = j[2];
        v19 = v11;
        if ( (*(_BYTE *)(v11 + 170) & 0x60) == 0 )
        {
          v17 = v11;
          if ( *(_BYTE *)(v11 + 177) != 5 )
          {
            v12 = sub_91CF50((__int64)v20, &v19);
            if ( v13 )
            {
              v14 = v12 || (int *)v13 == &v21 || v17 < *(_QWORD *)(v13 + 32);
              v16 = v13;
              v18 = v14;
              v15 = sub_22077B0(40);
              *(_QWORD *)(v15 + 32) = v19;
              sub_220F040(v18, v15, v16, &v21);
              ++v25;
            }
          }
        }
      }
    }
    goto LABEL_3;
  }
  return sub_127AA60(v3);
}
