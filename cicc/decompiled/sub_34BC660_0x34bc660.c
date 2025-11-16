// Function: sub_34BC660
// Address: 0x34bc660
//
void __fastcall sub_34BC660(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rbx
  __int64 i; // r12
  __int64 (*v4)(); // rax
  __int64 v5; // rax

  v1 = a1 + 320;
  v2 = *(_QWORD *)(a1 + 328);
  if ( v2 != a1 + 320 )
  {
    do
    {
      while ( !*(_BYTE *)(v2 + 260) || !*(_BYTE *)(v2 + 216) )
      {
        v2 = *(_QWORD *)(v2 + 8);
        if ( v1 == v2 )
          return;
      }
      for ( i = *(_QWORD *)(v2 + 56); *(_WORD *)(i + 68) != 4; i = *(_QWORD *)(i + 8) )
      {
        while ( (*(_BYTE *)i & 4) != 0 )
        {
          i = *(_QWORD *)(i + 8);
          if ( *(_WORD *)(i + 68) == 4 )
            goto LABEL_11;
        }
        while ( (*(_BYTE *)(i + 44) & 8) != 0 )
          i = *(_QWORD *)(i + 8);
      }
LABEL_11:
      v4 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 128LL);
      if ( v4 == sub_2DAC790 )
        BUG();
      v5 = v4();
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v5 + 888LL))(v5, v2, i);
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v1 != v2 );
  }
}
