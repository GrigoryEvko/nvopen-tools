// Function: sub_3574560
// Address: 0x3574560
//
bool __fastcall sub_3574560(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // rbx
  unsigned __int8 (__fastcall *v5)(__int64); // r12
  unsigned __int16 v7; // ax
  int v8[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = a1[3];
  v3 = *a1;
  v4 = a1[1];
  v5 = (unsigned __int8 (__fastcall *)(__int64))a1[2];
  while ( 1 )
  {
    if ( v3 == v2 )
      return 0;
LABEL_3:
    if ( !*(_BYTE *)v3 )
    {
      v8[0] = *(_DWORD *)(v3 + 8);
      v7 = sub_3574380(a2, v8);
      if ( HIBYTE(v7) )
      {
        if ( !(_BYTE)v7 )
          return v3 != v2;
      }
    }
    while ( 1 )
    {
      v3 += 40;
      if ( v4 == v3 )
        break;
      if ( v5(v3) )
      {
        if ( v3 != v2 )
          goto LABEL_3;
        return 0;
      }
    }
  }
}
