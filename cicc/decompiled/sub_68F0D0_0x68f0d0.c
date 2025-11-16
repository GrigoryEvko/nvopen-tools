// Function: sub_68F0D0
// Address: 0x68f0d0
//
_DWORD *__fastcall sub_68F0D0(
        __int64 a1,
        _QWORD *a2,
        int a3,
        int a4,
        unsigned int a5,
        _DWORD *a6,
        _QWORD *a7,
        _QWORD *a8,
        _DWORD *a9)
{
  int v11; // eax
  int v12; // ebx
  __int64 v13; // rax
  _DWORD *result; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  char v20; // al
  __int64 v21; // rax
  char i; // dl
  __int64 v23; // rdi
  char v24; // dl
  __int64 v25; // rax
  char v26; // al
  __int64 v27; // rax
  char j; // dl
  __int64 v30; // [rsp+8h] [rbp-38h]

  *a9 = 0;
  v11 = 1;
  if ( !a3 )
    v11 = a4;
  v12 = v11;
  if ( !dword_4D04474 )
    v12 = 0;
  v30 = sub_8D46C0(a1);
  if ( dword_4F04C44 == -1 )
  {
    v13 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v13 + 6) & 6) == 0 && *(_BYTE *)(v13 + 4) != 12 )
    {
      if ( *((_BYTE *)a2 + 16) != 3 )
        goto LABEL_14;
LABEL_11:
      *a7 = a1;
      result = (_DWORD *)*a2;
      *a8 = *a2;
      return result;
    }
  }
  if ( (unsigned int)sub_8DBE70(a1) || (unsigned int)sub_8DBE70(*a2) )
  {
    sub_6F4200(a2, a1, a5, 0);
    *a9 = 1;
    return a9;
  }
  if ( *((_BYTE *)a2 + 16) == 3 )
    goto LABEL_11;
LABEL_14:
  v15 = *a2;
  if ( (unsigned int)sub_8D2600(*a2) )
  {
LABEL_15:
    if ( (unsigned int)sub_6E5430(v15, a2, v16, v17, v18, v19) )
      sub_6851C0(0xABu, a6);
    sub_6E6840(a2);
    *a9 = 1;
    return a9;
  }
  if ( !(unsigned int)sub_8D2310(*a2) )
  {
LABEL_35:
    v26 = *((_BYTE *)a2 + 17);
    if ( v26 != 1 )
      goto LABEL_36;
    goto LABEL_39;
  }
  v20 = *((_BYTE *)a2 + 17);
  if ( v20 == 3 )
  {
    v15 = v30;
    if ( (unsigned int)sub_8D2E30(v30) )
      goto LABEL_15;
    goto LABEL_35;
  }
  if ( v20 != 1 )
    goto LABEL_21;
LABEL_39:
  if ( !(unsigned int)sub_6ED0A0(a2) )
    goto LABEL_28;
  v26 = *((_BYTE *)a2 + 17);
LABEL_36:
  if ( v26 != 3 )
  {
LABEL_21:
    if ( (unsigned int)sub_6ED0A0(a2) )
    {
      if ( !v12 && *((_BYTE *)a2 + 16) )
      {
        v21 = *a2;
        for ( i = *(_BYTE *)(*a2 + 140LL); i == 12; i = *(_BYTE *)(v21 + 140) )
          v21 = *(_QWORD *)(v21 + 160);
        if ( i )
        {
          sub_6E68E0(126, a2);
          *a9 = 1;
        }
      }
    }
    else if ( unk_4D0435C | (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C) && (unsigned int)sub_8D3A70(*a2) )
    {
      sub_6FA340(a2);
    }
    else if ( !a3 && *((_BYTE *)a2 + 16) )
    {
      v27 = *a2;
      for ( j = *(_BYTE *)(*a2 + 140LL); j == 12; j = *(_BYTE *)(v27 + 140) )
        v27 = *(_QWORD *)(v27 + 160);
      if ( j )
      {
        sub_6E68E0(v12 == 0 ? 126 : 2462, a2);
        *a9 = 1;
      }
    }
  }
LABEL_28:
  v23 = v30;
  *a7 = sub_72D2E0(v30, 0);
  if ( !*((_BYTE *)a2 + 16) )
    goto LABEL_33;
  v23 = *a2;
  v24 = *(_BYTE *)(*a2 + 140LL);
  if ( v24 == 12 )
  {
    v25 = *a2;
    do
    {
      v25 = *(_QWORD *)(v25 + 160);
      v24 = *(_BYTE *)(v25 + 140);
    }
    while ( v24 == 12 );
  }
  if ( v24 )
  {
    result = (_DWORD *)sub_72D2E0(v23, 0);
    *a8 = result;
  }
  else
  {
LABEL_33:
    result = (_DWORD *)sub_72C930(v23);
    *a8 = result;
  }
  return result;
}
