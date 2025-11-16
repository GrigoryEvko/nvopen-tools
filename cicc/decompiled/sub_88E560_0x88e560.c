// Function: sub_88E560
// Address: 0x88e560
//
__int64 __fastcall sub_88E560(__int64 *a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v7; // r14
  char v8; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r14
  __int64 *v15; // rdi
  __int64 *v16; // rax
  unsigned int v17; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 *v18[5]; // [rsp+8h] [rbp-28h] BYREF

  v18[0] = a1;
  if ( a1 && *((_BYTE *)a1 + 8) == 3 )
  {
    sub_72F220(v18);
    a1 = v18[0];
  }
  v6 = 1;
  if ( !(unsigned int)sub_88D7A0((__int64)a1, (__int64)a2, a3, a4, a5, a6) )
  {
    v7 = v18[0][4];
    v8 = *(_BYTE *)(v7 + 173);
    if ( v8 == 12 || !v8 )
      return 1;
    v6 = sub_8D2780(*(_QWORD *)(v7 + 128));
    if ( !v6 )
    {
      if ( a2 )
        sub_6851C0(0xAA0u, a2);
      return v6;
    }
    v14 = sub_620FA0(v7, &v17);
    if ( v14 < 0 || (v6 = v17) != 0 )
    {
      if ( a2 )
        sub_6851C0(0xA9Fu, a2);
      return 0;
    }
    v15 = (__int64 *)*v18[0];
    v18[0] = v15;
    if ( v15 )
    {
      if ( *((_BYTE *)v15 + 8) != 3 || (sub_72F220(v18), (v15 = v18[0]) != 0) )
      {
        if ( !(unsigned int)sub_88D7A0((__int64)v15, (__int64)&v17, v10, v11, v12, v13) )
        {
          v16 = v18[0];
          if ( v14 )
          {
            while ( v16 )
            {
              v16 = (__int64 *)*v16;
              v18[0] = v16;
              if ( !v16 )
                break;
              if ( *((_BYTE *)v16 + 8) == 3 )
              {
                sub_72F220(v18);
                v16 = v18[0];
              }
              if ( !--v14 )
                goto LABEL_29;
            }
            goto LABEL_24;
          }
LABEL_29:
          if ( !v16 )
            goto LABEL_24;
        }
        return 1;
      }
    }
LABEL_24:
    if ( a2 )
    {
      sub_6851C0(0xB27u, a2);
      return v6;
    }
    return 0;
  }
  return v6;
}
