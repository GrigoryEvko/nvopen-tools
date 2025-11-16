// Function: sub_88E770
// Address: 0x88e770
//
__int64 __fastcall sub_88E770(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v8; // r12
  __int64 *v9; // rax
  int v10; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 *v11[3]; // [rsp+8h] [rbp-18h] BYREF

  v11[0] = a1;
  if ( a1 && *((_BYTE *)a1 + 8) == 3 )
  {
    sub_72F220(v11);
    a1 = v11[0];
  }
  v2 = a1[4];
  if ( !(unsigned int)sub_8D3D40(*(_QWORD *)(v2 + 128)) && !(unsigned int)sub_8D2780(*(_QWORD *)(v2 + 128)) )
    return sub_72C930();
  if ( (unsigned int)sub_88D7A0((__int64)v11[0], a2, v3, v4, v5, v6) )
    return dword_4D03B80;
  v8 = sub_620FA0(v2, &v10);
  if ( v8 < 0 || v10 )
    return sub_72C930();
  v9 = (__int64 *)*v11[0];
  v11[0] = v9;
  if ( !v9 )
  {
    if ( v8 )
      goto LABEL_17;
    return sub_72C930();
  }
  if ( *((_BYTE *)v9 + 8) == 3 )
  {
    sub_72F220(v11);
    v9 = v11[0];
    if ( v8 )
      goto LABEL_17;
LABEL_21:
    if ( !v9 )
      return sub_72C930();
  }
  else if ( v8 )
  {
LABEL_17:
    while ( v9 )
    {
      v9 = (__int64 *)*v9;
      v11[0] = v9;
      if ( !v9 )
        break;
      if ( *((_BYTE *)v9 + 8) == 3 )
      {
        sub_72F220(v11);
        v9 = v11[0];
      }
      if ( !--v8 )
        goto LABEL_21;
    }
    return sub_72C930();
  }
  return v9[4];
}
