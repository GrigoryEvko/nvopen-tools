// Function: sub_8B0460
// Address: 0x8b0460
//
__int64 __fastcall sub_8B0460(unsigned __int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  __int64 v7; // rbx
  unsigned __int16 v8; // ax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 result; // rax
  __int64 v16; // rcx
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __m128i **v23; // rax
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  unsigned int v26; // [rsp+1Ch] [rbp-44h]
  __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v28[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = a3;
  v7 = a1;
  v25 = a2;
  v27 = 0;
  if ( word_4F06418[0] == 160 )
  {
    v26 = 0;
    do
    {
      if ( !v6 )
        ++*(_DWORD *)(v7 + 168);
      ++*(_QWORD *)(v7 + 224);
      v28[0] = *(_QWORD *)&dword_4F063F8;
      sub_7B8B50(a1, (unsigned int *)a2, a3, (__int64)a4, a5, a6);
      if ( word_4F06418[0] == 43 )
      {
        sub_7B8B50(a1, (unsigned int *)a2, v9, v10, v11, v12);
        if ( word_4F06418[0] == 44 )
        {
          if ( v6 || (v16 = *(unsigned int *)(v7 + 92), (_DWORD)v16) )
          {
            a2 = (__int64)dword_4F07508;
            a1 = 994;
            sub_6851C0(0x3E2u, dword_4F07508);
            sub_7B8B50(0x3E2u, dword_4F07508, v19, v20, v21, v22);
            v8 = word_4F06418[0];
          }
          else
          {
            v17 = v26;
            ++*(_DWORD *)(v7 + 176);
            *(_DWORD *)(v7 + 24) = 1;
            if ( v26 )
            {
              a2 = (__int64)dword_4F07508;
              a1 = 788;
              sub_6851C0(0x314u, dword_4F07508);
              *(_DWORD *)(v7 + 52) = 1;
            }
            sub_7B8B50(a1, (unsigned int *)a2, v17, v16, v13, v14);
            v18 = sub_727300();
            a4 = qword_4F04C68;
            v18[3] = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184);
            v18[4] = v28[0];
            *v18 = *(_QWORD *)(v7 + 488);
            a3 = *(_QWORD *)(v7 + 192);
            *(_QWORD *)(v7 + 488) = v18;
            if ( a3 )
              *(_QWORD *)(a3 + 32) = v18;
            v8 = word_4F06418[0];
          }
        }
        else
        {
          a2 = (__int64)v28;
          sub_890230(v7, v28, &v27);
          a1 = v7;
          sub_8AF6E0(v7);
          v8 = word_4F06418[0];
          v26 = 1;
          if ( word_4F06418[0] == 294 )
          {
            a2 = *(unsigned int *)(v7 + 92);
            if ( (_DWORD)a2 )
              break;
            a1 = 0;
            v24 = *(_QWORD *)(v7 + 488);
            v23 = sub_6DE780(0, a2, v24, (__int64)a4);
            a3 = v24;
            *(_QWORD *)(v24 + 16) = v23;
            v8 = word_4F06418[0];
          }
        }
      }
      else
      {
        a2 = (__int64)dword_4F07508;
        a1 = 707;
        sub_6851C0(0x2C3u, dword_4F07508);
        v8 = word_4F06418[0];
      }
    }
    while ( v8 == 160 );
  }
  if ( v25 && !*(_DWORD *)(v7 + 24) )
  {
    if ( !v6 )
      ++*(_DWORD *)(v7 + 168);
    ++*(_QWORD *)(v7 + 224);
    sub_890230(v7, &dword_4F077C8, &v27);
    sub_897CB0(v7, v25);
    *(_BYTE *)(*(_QWORD *)v7 + 133LL) |= 8u;
  }
  result = v27;
  *(_QWORD *)(v7 + 192) = v27;
  if ( v6 )
  {
    if ( *(_QWORD *)(v7 + 224) > 1u )
    {
      result = sub_6851C0(0x309u, dword_4F07508);
      *(_DWORD *)(v7 + 52) = 1;
    }
  }
  return result;
}
