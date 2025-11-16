// Function: sub_729880
// Address: 0x729880
//
void __fastcall sub_729880(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        char a8,
        char a9,
        char a10,
        char a11,
        char a12,
        char a13)
{
  __int64 v13; // rax
  __int64 v14; // r10
  unsigned int v15; // edx
  char v16; // r13
  char v17; // r12
  __int64 v18; // rax
  __int64 v19; // rdx

  v13 = sub_727A60();
  v14 = v13;
  *a7 = v13;
  v15 = a3;
  *(_QWORD *)v13 = a4;
  v16 = *(_BYTE *)(v13 + 72);
  *(_QWORD *)(v13 + 8) = a5;
  *(_QWORD *)(v13 + 16) = a6;
  *(_DWORD *)(v13 + 24) = a2;
  v17 = v16 & 0x81
      | ((a13 & 1) << 6)
      | (2 * (a12 & 1))
      | (4 * (a8 & 1))
      | (8 * (a9 & 1))
      | (16 * (a10 & 1))
      | (32 * (a11 & 1));
  *(_DWORD *)(v13 + 32) = a3;
  *(_BYTE *)(v13 + 72) = v17;
  if ( a1 )
  {
    if ( *(_QWORD *)(a1 + 40) )
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 48) + 56LL) = v13;
      *(_QWORD *)(a1 + 48) = v13;
      if ( unk_4F07280 != a1 )
      {
LABEL_4:
        v15 = *(_DWORD *)(v13 + 32);
        a2 = *(_DWORD *)(v13 + 24);
        goto LABEL_5;
      }
    }
    else
    {
      *(_QWORD *)(a1 + 40) = v13;
      *(_QWORD *)(a1 + 48) = v13;
      if ( unk_4F07280 != a1 )
        goto LABEL_4;
    }
    *(_DWORD *)(a1 + 28) = -1;
    goto LABEL_4;
  }
  *(_BYTE *)(v13 + 72) = v17 | 0x80;
  v18 = unk_4F07280;
  if ( unk_4F07280 )
  {
    do
    {
      v19 = v18;
      v18 = *(_QWORD *)(v18 + 56);
    }
    while ( v18 );
    *(_QWORD *)(v19 + 56) = v14;
    v15 = *(_DWORD *)(v14 + 32);
    a2 = *(_DWORD *)(v14 + 24);
  }
  else
  {
    unk_4F07280 = v14;
  }
LABEL_5:
  sub_729230(v14, a2, v15);
}
