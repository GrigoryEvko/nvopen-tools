// Function: sub_25590F0
// Address: 0x25590f0
//
__int64 __fastcall sub_25590F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v9; // rax
  char v10; // dl
  _BYTE v11[64]; // [rsp+10h] [rbp-40h] BYREF

  v5 = *a1;
  if ( !*(_BYTE *)(v5 + 228) )
  {
    if ( sub_C8CA60(v5 + 200, a2) )
      return 1;
    v5 = *a1;
    goto LABEL_9;
  }
  v6 = *(_QWORD **)(v5 + 208);
  v7 = (__int64)&v6[*(unsigned int *)(v5 + 220)];
  if ( v6 == (_QWORD *)v7 )
  {
LABEL_9:
    if ( !(unsigned __int8)sub_B19060(v5 + 104, a2, v7, a4) && (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 1 )
    {
      v9 = sub_2559000(*a1, a1[1], *(_QWORD *)(a2 - 96), a2);
      if ( v10 )
      {
        if ( v9 )
          sub_BED950((__int64)v11, *a1 + 200, a2);
      }
    }
    return 1;
  }
  while ( a2 != *v6 )
  {
    if ( (_QWORD *)v7 == ++v6 )
      goto LABEL_9;
  }
  return 1;
}
