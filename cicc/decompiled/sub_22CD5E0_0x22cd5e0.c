// Function: sub_22CD5E0
// Address: 0x22cd5e0
//
__int64 __fastcall sub_22CD5E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // eax
  unsigned int v5; // eax
  __int64 v8; // r15
  bool v10; // zf
  __int64 v11; // [rsp+10h] [rbp-A0h]
  __int64 v12; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v13; // [rsp+28h] [rbp-88h]
  unsigned int v14; // [rsp+30h] [rbp-80h]
  unsigned __int64 v15; // [rsp+38h] [rbp-78h]
  unsigned int v16; // [rsp+40h] [rbp-70h]
  unsigned __int8 v17[8]; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v18; // [rsp+58h] [rbp-58h]
  unsigned int v19; // [rsp+60h] [rbp-50h]
  unsigned __int64 v20; // [rsp+68h] [rbp-48h]
  unsigned int v21; // [rsp+70h] [rbp-40h]
  char v22; // [rsp+78h] [rbp-38h]

  v4 = *(_DWORD *)(a3 + 4);
  v12 = 0;
  v5 = v4 & 0x7FFFFFF;
  if ( v5 )
  {
    v8 = 0;
    v11 = 8LL * v5;
    while ( 1 )
    {
      sub_22CCF60(
        (__int64)v17,
        a2,
        *(_QWORD *)(*(_QWORD *)(a3 - 8) + 4 * v8),
        *(_QWORD *)(*(_QWORD *)(a3 - 8) + 32LL * *(unsigned int *)(a3 + 72) + v8),
        a4,
        a3);
      if ( !v22 )
        break;
      sub_22C0C70((__int64)&v12, (__int64)v17, 0, 0, 1u);
      if ( (_BYTE)v12 == 6 )
      {
        sub_22C0650(a1, (unsigned __int8 *)&v12);
        v10 = v22 == 0;
        *(_BYTE *)(a1 + 40) = 1;
        if ( !v10 )
        {
          v22 = 0;
          sub_22C0090(v17);
        }
        goto LABEL_15;
      }
      if ( !v22 )
        goto LABEL_3;
      v22 = 0;
      if ( (unsigned int)v17[0] - 4 > 1 )
        goto LABEL_3;
      if ( v21 > 0x40 && v20 )
        j_j___libc_free_0_0(v20);
      if ( v19 > 0x40 && v18 )
      {
        j_j___libc_free_0_0(v18);
        v8 += 8;
        if ( v8 == v11 )
          goto LABEL_14;
      }
      else
      {
LABEL_3:
        v8 += 8;
        if ( v8 == v11 )
          goto LABEL_14;
      }
    }
    *(_BYTE *)(a1 + 40) = 0;
    if ( (unsigned int)(unsigned __int8)v12 - 4 > 1 )
      return a1;
    goto LABEL_18;
  }
LABEL_14:
  sub_22C0650(a1, (unsigned __int8 *)&v12);
  *(_BYTE *)(a1 + 40) = 1;
LABEL_15:
  if ( (unsigned int)(unsigned __int8)v12 - 4 <= 1 )
  {
LABEL_18:
    if ( v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    if ( v14 > 0x40 && v13 )
      j_j___libc_free_0_0(v13);
  }
  return a1;
}
