// Function: sub_174A3D0
// Address: 0x174a3d0
//
__int64 __fastcall sub_174A3D0(_BYTE *a1)
{
  unsigned __int8 v1; // cl
  __int64 result; // rax
  __int64 v3; // r13
  __int64 *v4; // r12
  unsigned int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // [rsp+0h] [rbp-40h]
  int v9; // [rsp+Ch] [rbp-34h]

  v1 = a1[16];
  if ( v1 > 0x17u )
  {
    if ( v1 == 68 )
      return **((_QWORD **)a1 - 3);
    return *(_QWORD *)a1;
  }
  if ( v1 == 14 )
  {
    result = sub_174A310((__int64)a1);
    if ( result )
      return result;
    v1 = a1[16];
  }
  result = *(_QWORD *)a1;
  if ( v1 <= 0x10u && *(_BYTE *)(result + 8) == 16 )
  {
    v3 = *(_QWORD *)(result + 32);
    v4 = 0;
    if ( (_DWORD)v3 )
    {
      v5 = 0;
      while ( 1 )
      {
        v6 = sub_15A0A60((__int64)a1, v5);
        if ( !v6 )
          break;
        if ( *(_BYTE *)(v6 + 16) != 14 )
          break;
        v7 = sub_174A310(v6);
        if ( !v7 )
          break;
        if ( v4 )
        {
          v8 = (__int64 *)v7;
          v9 = sub_16431F0(v7);
          if ( v9 > (int)sub_16431F0((__int64)v4) )
            v4 = v8;
        }
        else
        {
          v4 = (__int64 *)v7;
        }
        if ( (_DWORD)v3 == ++v5 )
          goto LABEL_19;
      }
      return *(_QWORD *)a1;
    }
LABEL_19:
    result = (__int64)sub_16463B0(v4, v3);
    if ( !result )
      return *(_QWORD *)a1;
  }
  return result;
}
