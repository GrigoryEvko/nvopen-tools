// Function: sub_F7DD30
// Address: 0xf7dd30
//
_QWORD *__fastcall sub_F7DD30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  char v6; // dl
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rdx
  unsigned int v11; // esi
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  _QWORD v17[2]; // [rsp+0h] [rbp-90h] BYREF
  const char *v18; // [rsp+10h] [rbp-80h]
  __int16 v19; // [rsp+20h] [rbp-70h]
  _QWORD v20[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v21; // [rsp+50h] [rbp-40h]

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 14 )
  {
    v21 = 259;
    v20[0] = "scevgep";
    return sub_F7CA10((__int64 *)(a1 + 520), a2, a3, (__int64)v20, 0);
  }
  else
  {
    v6 = **(_BYTE **)(a1 + 16);
    if ( a5 )
    {
      if ( v6 )
      {
        v17[0] = *(_QWORD *)(a1 + 16);
        v18 = ".iv.next";
        v19 = 771;
      }
      else
      {
        v17[0] = ".iv.next";
        v19 = 259;
      }
      v7 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 600)
                                                                                                + 32LL))(
             *(_QWORD *)(a1 + 600),
             15,
             a2,
             a3,
             0,
             0,
             v17[0]);
      if ( !v7 )
      {
        v21 = 257;
        v7 = sub_B504D0(15, a2, a3, (__int64)v20, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
          *(_QWORD *)(a1 + 608),
          v7,
          v17,
          *(_QWORD *)(a1 + 576),
          *(_QWORD *)(a1 + 584));
        v8 = *(_QWORD *)(a1 + 520);
        v9 = v8 + 16LL * *(unsigned int *)(a1 + 528);
        while ( v9 != v8 )
        {
          v10 = *(_QWORD *)(v8 + 8);
          v11 = *(_DWORD *)v8;
          v8 += 16;
          sub_B99FD0(v7, v11, v10);
        }
      }
    }
    else
    {
      if ( v6 )
      {
        v17[0] = *(_QWORD *)(a1 + 16);
        v18 = ".iv.next";
        v19 = 771;
      }
      else
      {
        v17[0] = ".iv.next";
        v19 = 259;
      }
      v7 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 600)
                                                                                                + 32LL))(
             *(_QWORD *)(a1 + 600),
             13,
             a2,
             a3,
             0,
             0,
             v17[0]);
      if ( !v7 )
      {
        v21 = 257;
        v7 = sub_B504D0(13, a2, a3, (__int64)v20, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
          *(_QWORD *)(a1 + 608),
          v7,
          v17,
          *(_QWORD *)(a1 + 576),
          *(_QWORD *)(a1 + 584));
        v12 = *(_QWORD *)(a1 + 520);
        v13 = v12 + 16LL * *(unsigned int *)(a1 + 528);
        while ( v13 != v12 )
        {
          v14 = *(_QWORD *)(v12 + 8);
          v15 = *(_DWORD *)v12;
          v12 += 16;
          sub_B99FD0(v7, v15, v14);
        }
      }
    }
  }
  return (_QWORD *)v7;
}
