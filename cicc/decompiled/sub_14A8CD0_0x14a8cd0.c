// Function: sub_14A8CD0
// Address: 0x14a8cd0
//
__int64 __fastcall sub_14A8CD0(__int64 ***a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned int v3; // eax
  unsigned int v4; // r14d
  __int64 *v5; // r13
  __int64 v6; // rbx
  __int64 *i; // [rsp+8h] [rbp-48h]
  _QWORD v9[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a2 + 40);
  v9[1] = *(_QWORD *)(a2 - 48);
  v9[0] = v2;
  v3 = sub_15CC350(v9);
  if ( (_BYTE)v3 )
  {
    v4 = v3;
    v5 = **a1;
    for ( i = &v5[*((unsigned int *)*a1 + 2)]; i != v5; ++v5 )
    {
      v6 = *v5;
      if ( !(unsigned __int8)sub_15CCD40(a1[1], v9, *(_QWORD *)(*v5 + 40)) )
      {
        while ( 1 )
        {
          v6 = *(_QWORD *)(v6 + 8);
          if ( !v6 )
            break;
          if ( !(unsigned __int8)sub_15CCFD0(a1[1], v9, v6) )
            return 0;
        }
      }
    }
  }
  else
  {
    return 0;
  }
  return v4;
}
