// Function: sub_22CDEF0
// Address: 0x22cdef0
//
__int64 __fastcall sub_22CDEF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v7; // al
  unsigned int v8; // eax
  unsigned int v9; // eax
  bool v11; // zf
  unsigned int v12; // eax
  unsigned __int8 v14[8]; // [rsp+10h] [rbp-90h] BYREF
  const void *v15; // [rsp+18h] [rbp-88h] BYREF
  unsigned int v16; // [rsp+20h] [rbp-80h]
  const void *v17; // [rsp+28h] [rbp-78h] BYREF
  unsigned int v18; // [rsp+30h] [rbp-70h]
  char v19; // [rsp+38h] [rbp-68h]
  unsigned __int8 v20[40]; // [rsp+40h] [rbp-60h] BYREF
  char v21; // [rsp+68h] [rbp-38h]

  sub_22C7100((__int64)v14, a2, a3, a4, a5);
  if ( !v19 )
  {
    sub_22CDAD0(a2);
    sub_22C7100((__int64)v20, a2, a3, a4, a5);
    if ( v19 )
    {
      if ( v21 )
      {
        sub_22C0090(v14);
        sub_22C0650((__int64)v14, v20);
      }
      else
      {
        v19 = 0;
        sub_22C0090(v14);
      }
    }
    else
    {
      if ( !v21 )
        goto LABEL_2;
      sub_22C0650((__int64)v14, v20);
      v19 = 1;
    }
    if ( v21 )
    {
      v21 = 0;
      sub_22C0090(v20);
    }
  }
LABEL_2:
  v7 = v14[0];
  *(_WORD *)a1 = v14[0];
  if ( v7 > 3u )
  {
    if ( (unsigned __int8)(v7 - 4) > 1u )
      goto LABEL_8;
    v8 = v16;
    *(_DWORD *)(a1 + 16) = v16;
    if ( v8 > 0x40 )
    {
      sub_C43780(a1 + 8, &v15);
      v12 = v18;
      *(_DWORD *)(a1 + 32) = v18;
      if ( v12 <= 0x40 )
        goto LABEL_6;
    }
    else
    {
      *(_QWORD *)(a1 + 8) = v15;
      v9 = v18;
      *(_DWORD *)(a1 + 32) = v18;
      if ( v9 <= 0x40 )
      {
LABEL_6:
        *(_QWORD *)(a1 + 24) = v17;
LABEL_7:
        *(_BYTE *)(a1 + 1) = v14[1];
        goto LABEL_8;
      }
    }
    sub_C43780(a1 + 24, &v17);
    goto LABEL_7;
  }
  if ( v7 > 1u )
  {
    v11 = v19 == 0;
    *(_QWORD *)(a1 + 8) = v15;
    if ( v11 )
      return a1;
LABEL_12:
    v19 = 0;
    sub_22C0090(v14);
    return a1;
  }
LABEL_8:
  if ( v19 )
    goto LABEL_12;
  return a1;
}
