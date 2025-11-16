// Function: sub_8D0910
// Address: 0x8d0910
//
void __fastcall sub_8D0910(_QWORD *a1)
{
  void *v2; // rdi
  _DWORD *v3; // rcx
  _QWORD *v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned int v10; // edx
  unsigned int v11; // eax

  v2 = (void *)qword_4D03FF0;
  if ( (_QWORD *)qword_4D03FF0 != a1 )
  {
    sub_8D06C0((_QWORD *)qword_4D03FF0);
    qword_4D03FF0 = (__int64)a1;
    v4 = (_QWORD *)qword_4F60540;
    v5 = a1[2];
    if ( qword_4F60540 )
    {
      do
      {
        v2 = (void *)v4[1];
        v3 = memcpy(v2, (const void *)(v5 + v4[3]), v4[2]);
        v6 = v4[4];
        if ( v6 )
          *(_QWORD *)((char *)a1 + v6) = v3;
        v4 = (_QWORD *)*v4;
      }
      while ( v4 );
    }
    qword_4F07288 = a1[1];
    v7 = (unsigned int)dword_4F04C64;
    unk_4F07290 = a1[25];
    qword_4F072C0 = a1[31];
    qword_4F07300 = a1[39];
    if ( dword_4F04C64 != -1 )
    {
      v8 = qword_4F04C68[0];
      v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( v9 )
      {
        v2 = (void *)0xA3A0FD5C5F02A3A1LL;
        while ( 1 )
        {
          v3 = *(_DWORD **)(v9 + 184);
          if ( v3 )
            v3[60] = 1594008481 * ((v9 - v8) >> 3);
          if ( !*(_BYTE *)(v9 + 4) )
            break;
          v9 -= 776;
        }
      }
      v10 = 0;
      if ( unk_4F04C48 != -1 && (*(_BYTE *)(v8 + 776LL * unk_4F04C48 + 10) & 1) != 0 )
      {
        if ( dword_4D047C8 )
        {
          v11 = sub_7D3BE0(v2, v8, 0, v3, v7);
          LODWORD(v7) = dword_4F04C64;
          v10 = v11;
        }
      }
      sub_85FE80(v7, 1, v10);
    }
  }
}
