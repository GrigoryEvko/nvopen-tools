// Function: sub_2A856E0
// Address: 0x2a856e0
//
void __fastcall sub_2A856E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned __int64 v6; // r13
  __int64 *v7; // rbx
  __int64 *v8; // r14
  __int64 v9; // rsi
  unsigned __int64 *v10; // r12
  unsigned __int64 v11; // [rsp+8h] [rbp-48h]
  _BYTE v12[49]; // [rsp+1Fh] [rbp-31h] BYREF

  v11 = qword_500C090 - qword_500C088;
  if ( qword_500C090 != qword_500C088 )
  {
    if ( v11 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1, a2, a3);
    v3 = sub_22077B0(v11);
    v4 = qword_500C090;
    v5 = qword_500C088;
    v6 = v3;
    if ( qword_500C090 != qword_500C088 )
    {
      v7 = (__int64 *)v3;
      do
      {
        if ( v7 )
        {
          *v7 = (__int64)(v7 + 2);
          sub_2A80060(v7, *(_BYTE **)v5, *(_QWORD *)v5 + *(_QWORD *)(v5 + 8));
        }
        v5 += 32;
        v7 += 4;
      }
      while ( v4 != v5 );
      if ( v7 == (__int64 *)v6 )
      {
LABEL_16:
        j_j___libc_free_0(v6);
        return;
      }
      v8 = (__int64 *)v6;
      do
      {
        v9 = (__int64)v8;
        v8 += 4;
        sub_2A85510((__int64)v12, v9, a1);
      }
      while ( v7 != v8 );
      v10 = (unsigned __int64 *)v6;
      do
      {
        if ( (unsigned __int64 *)*v10 != v10 + 2 )
          j_j___libc_free_0(*v10);
        v10 += 4;
      }
      while ( v7 != (__int64 *)v10 );
    }
    if ( !v6 )
      return;
    goto LABEL_16;
  }
}
