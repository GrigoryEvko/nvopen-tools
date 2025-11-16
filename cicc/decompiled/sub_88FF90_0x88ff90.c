// Function: sub_88FF90
// Address: 0x88ff90
//
void __fastcall sub_88FF90(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rax

  if ( a2 )
  {
    v2 = a2;
    do
    {
      while ( (*((_BYTE *)v2 + 33) & 1) == 0 )
      {
        v2 = (__int64 *)*v2;
        if ( !v2 )
          return;
      }
      v3 = qword_4F601A0;
      if ( qword_4F601A0 )
        qword_4F601A0 = *(_QWORD *)qword_4F601A0;
      else
        v3 = sub_823970(32);
      *(_DWORD *)(v3 + 24) = 0;
      *(_QWORD *)v3 = 0;
      *(_QWORD *)(v3 + 8) = v2;
      *(_QWORD *)(v3 + 16) = v2;
      *(_DWORD *)(v3 + 24) = *(_DWORD *)(a1 + 72);
      if ( *(_QWORD *)a1 )
      {
        *(_QWORD *)v3 = **(_QWORD **)(a1 + 8);
        **(_QWORD **)(a1 + 8) = v3;
      }
      else
      {
        *(_QWORD *)a1 = v3;
      }
      *(_QWORD *)(a1 + 8) = v3;
      v2 = (__int64 *)*v2;
    }
    while ( v2 );
  }
}
