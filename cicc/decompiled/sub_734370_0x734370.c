// Function: sub_734370
// Address: 0x734370
//
void __fastcall sub_734370(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rsi

  v1 = a1[3];
  a1[5] = 0;
  if ( v1 )
  {
    v3 = *(_QWORD *)(v1 + 32);
    if ( v3 )
    {
      v4 = *(_QWORD *)(v3 + 32);
      if ( v4 )
      {
        v5 = *(_QWORD *)(v4 + 32);
        if ( v5 )
        {
          sub_7342B0(v5);
          *(_QWORD *)(v4 + 32) = 0;
        }
        *(_QWORD *)(v4 + 24) = 0;
        sub_7340D0(v4, 0, 0);
        *(_QWORD *)(v3 + 32) = 0;
      }
      *(_QWORD *)(v3 + 24) = 0;
      sub_7340D0(v3, 0, 0);
      *(_QWORD *)(v1 + 32) = 0;
    }
    *(_QWORD *)(v1 + 24) = 0;
    sub_7340D0(v1, 0, 0);
    a1[3] = 0;
  }
  v6 = a1[6];
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 56);
    v8 = qword_4F06BC0;
    if ( v7 )
    {
      while ( 1 )
      {
        *(_QWORD *)(v6 + 32) = v8;
        v6 = v7;
        if ( !*(_QWORD *)(v7 + 56) )
          break;
        v7 = *(_QWORD *)(v7 + 56);
      }
    }
    else
    {
      v7 = a1[6];
    }
    *(_QWORD *)(v7 + 32) = v8;
    *(_QWORD *)(v7 + 56) = *(_QWORD *)(v8 + 48);
    *(_QWORD *)(v8 + 48) = a1[6];
    a1[6] = 0;
  }
}
