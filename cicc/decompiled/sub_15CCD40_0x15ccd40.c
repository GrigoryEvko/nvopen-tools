// Function: sub_15CCD40
// Address: 0x15ccd40
//
__int64 __fastcall sub_15CCD40(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // r14
  unsigned int v5; // eax
  unsigned int v6; // r12d
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // [rsp+Ch] [rbp-34h]

  v3 = a2[1];
  v4 = *a2;
  LOBYTE(v5) = sub_15CC8F0(a1, v3, a3);
  if ( !(_BYTE)v5 )
    return 0;
  v6 = v5;
  if ( !sub_157F0B0(v3) )
  {
    v8 = *(_QWORD *)(v3 + 8);
    if ( v8 )
    {
      while ( 1 )
      {
        v9 = sub_1648700(v8);
        if ( (unsigned __int8)(*(_BYTE *)(v9 + 16) - 25) <= 9u )
          break;
        v8 = *(_QWORD *)(v8 + 8);
        if ( !v8 )
          return v6;
      }
      v11 = 0;
LABEL_11:
      v10 = *(_QWORD *)(v9 + 40);
      if ( v4 != v10 )
      {
        if ( sub_15CC8F0(a1, v3, v10) )
          goto LABEL_9;
        return 0;
      }
      if ( v11 )
        return 0;
      v11 = 1;
LABEL_9:
      while ( 1 )
      {
        v8 = *(_QWORD *)(v8 + 8);
        if ( !v8 )
          break;
        v9 = sub_1648700(v8);
        if ( (unsigned __int8)(*(_BYTE *)(v9 + 16) - 25) <= 9u )
          goto LABEL_11;
      }
    }
  }
  return v6;
}
