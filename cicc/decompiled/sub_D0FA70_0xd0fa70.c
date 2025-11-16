// Function: sub_D0FA70
// Address: 0xd0fa70
//
void __fastcall sub_D0FA70(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r14
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax

  v1 = *(_QWORD *)(a1 + 64);
  if ( v1 )
  {
    *(_DWORD *)(v1 + 40) = 0;
    v2 = *(_QWORD **)(a1 + 64);
    if ( v2 )
    {
      v3 = v2[3];
      v4 = v2[2];
      if ( v3 == v4 )
      {
        if ( !v4 )
        {
LABEL_6:
          j_j___libc_free_0(v2, 48);
          goto LABEL_7;
        }
      }
      else
      {
        do
        {
          while ( !*(_BYTE *)(v4 + 24) )
          {
            v4 += 40;
            if ( v3 == v4 )
              goto LABEL_14;
          }
          v5 = *(_QWORD *)(v4 + 16);
          *(_BYTE *)(v4 + 24) = 0;
          if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
            sub_BD60C0((_QWORD *)v4);
          v4 += 40;
        }
        while ( v3 != v4 );
LABEL_14:
        v4 = v2[2];
        if ( !v4 )
          goto LABEL_6;
      }
      j_j___libc_free_0(v4, v2[4] - v4);
      goto LABEL_6;
    }
  }
LABEL_7:
  sub_D0EF00(*(_QWORD **)(a1 + 24));
}
