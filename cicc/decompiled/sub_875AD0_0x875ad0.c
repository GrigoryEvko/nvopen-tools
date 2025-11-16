// Function: sub_875AD0
// Address: 0x875ad0
//
void __fastcall sub_875AD0(__int64 a1, _DWORD *a2)
{
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 *v6; // rbx
  unsigned __int8 v7; // r15
  char v8; // di
  int v9; // r10d
  __int64 v10; // rcx
  FILE *v11; // rsi
  _DWORD *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rax
  unsigned int v16; // [rsp-54h] [rbp-54h]
  unsigned int v17; // [rsp-50h] [rbp-50h]
  unsigned int v18; // [rsp-4Ch] [rbp-4Ch]
  __m128i v19[4]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(char *)(a1 + 90) < 0 && !(unsigned int)sub_867A10() )
  {
    if ( qword_4F04C50 )
    {
      v4 = *(_QWORD *)(qword_4F04C50 + 32LL);
      if ( v4 )
      {
        if ( (*(_BYTE *)(v4 + 198) & 0x10) != 0 )
        {
          v5 = *(_QWORD *)a1;
          v6 = sub_736C60(21, *(__int64 **)(a1 + 104));
          if ( v6 )
          {
            v17 = 3263;
            v7 = 8;
            v8 = 21;
            v9 = 3286;
            v16 = 3285;
          }
          else
          {
            v7 = 5;
            v15 = sub_736C60(6, *(__int64 **)(a1 + 104));
            v8 = 6;
            v17 = 3287;
            v9 = 1444;
            v16 = 1215;
            v6 = v15;
          }
          v18 = v9;
          v10 = sub_5CFC90(v8, a1);
          if ( v10 )
          {
            v11 = (FILE *)v18;
            v12 = sub_67DF10(v7, v18, a2, v10, v5);
          }
          else
          {
            v11 = (FILE *)v16;
            v12 = sub_67DE50(v7, v16, a2, v5);
          }
          v13 = *((unsigned int *)v6 + 14);
          v14 = (unsigned int)dword_4F077C8;
          if ( (_DWORD)v13 == dword_4F077C8 )
          {
            v13 = *((unsigned __int16 *)v6 + 30);
            v14 = word_4F077CC[0];
          }
          if ( v13 != v14 )
          {
            v19[0] = 0u;
            sub_6855B0(v17, (FILE *)(v6 + 7), v19);
            v11 = (FILE *)v19;
            sub_67E370((__int64)v12, v19);
          }
          sub_685910((__int64)v12, v11);
        }
      }
    }
  }
}
