// Function: sub_291A050
// Address: 0x291a050
//
__int64 __fastcall sub_291A050(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v5; // rcx
  char v6; // r13
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r15
  char v9; // al
  __int64 v10; // r14
  __int64 v11; // rax
  char v12; // [rsp+Fh] [rbp-41h]

  while ( 1 )
  {
    v3 = *(unsigned __int8 *)(a2 + 8);
    if ( (unsigned __int8)v3 <= 3u )
      break;
    if ( (_BYTE)v3 == 5 )
      break;
    if ( (unsigned __int8)v3 <= 0x14u )
    {
      v5 = 1463376;
      if ( _bittest64(&v5, v3) )
        break;
    }
    v6 = sub_AE5020(a1, a2);
    v7 = (((unsigned __int64)(sub_9208B0(a1, a2) + 7) >> 3) + (1LL << v6) - 1) >> v6 << v6;
    v8 = sub_9208B0(a1, a2);
    v9 = *(_BYTE *)(a2 + 8);
    if ( v9 == 16 )
    {
      v10 = *(_QWORD *)(a2 + 24);
    }
    else
    {
      if ( v9 != 15 )
        return a2;
      v11 = sub_AE4AC0(a1, a2);
      v10 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL * (unsigned int)sub_AE1C80(v11, 0));
    }
    v12 = sub_AE5020(a1, v10);
    if ( v7 > (((unsigned __int64)(sub_9208B0(a1, v10) + 7) >> 3) + (1LL << v12) - 1) >> v12 << v12
      || v8 > sub_9208B0(a1, v10) )
    {
      break;
    }
    a2 = v10;
  }
  return a2;
}
