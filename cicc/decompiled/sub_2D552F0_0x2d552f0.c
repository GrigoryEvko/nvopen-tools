// Function: sub_2D552F0
// Address: 0x2d552f0
//
__int64 __fastcall sub_2D552F0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v4; // r15
  unsigned int v5; // r13d
  __int64 v6; // r14
  unsigned int v7; // r13d
  __int64 *v9; // [rsp+8h] [rbp-A8h]
  __int64 *v10; // [rsp+10h] [rbp-A0h]
  __int64 v11; // [rsp+18h] [rbp-98h]
  _QWORD v12[4]; // [rsp+20h] [rbp-90h] BYREF
  int v13; // [rsp+40h] [rbp-70h]
  char v14; // [rsp+44h] [rbp-6Ch]
  void *v15; // [rsp+50h] [rbp-60h] BYREF
  __int16 v16; // [rsp+70h] [rbp-40h]

  v12[0] = a3;
  memset(&v12[1], 0, 24);
  v13 = 1;
  v14 = 1;
  v9 = &a1[a2];
  if ( v9 == a1 )
  {
    return 0;
  }
  else
  {
    v10 = a1;
    v3 = 0;
    do
    {
      v4 = *v10;
      v5 = *(_DWORD *)(*v10 + 88);
      if ( v5 )
      {
        v6 = 0;
        v11 = v5;
        do
        {
          v7 = v6 + 1;
          if ( (_DWORD)v6 == -1
            || *(_QWORD *)(v4 - 32 + -32 - 32LL * *(unsigned int *)(v4 + 88)) == *(_QWORD *)(v4
                                                                                           - 32
                                                                                           + 32
                                                                                           * (v6
                                                                                            - *(unsigned int *)(v4 + 88)))
            || (unsigned __int8)sub_D0E970(v4, v7, 1u) )
          {
            v16 = 257;
            if ( sub_F44160(v4, v7, (__int64)v12, &v15) )
              v3 = 1;
          }
          ++v6;
        }
        while ( v11 != v6 );
      }
      ++v10;
    }
    while ( v9 != v10 );
  }
  return v3;
}
