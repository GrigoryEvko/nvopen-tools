// Function: sub_2973B00
// Address: 0x2973b00
//
__int64 __fastcall sub_2973B00(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  _QWORD *v4; // rsi
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r8
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_4F8662C);
  if ( LOBYTE(qword_4F8D3A0[17]) )
    sub_BB9660(a2, (__int64)&unk_4F8144C);
  sub_BB9660(a2, (__int64)&unk_4F89C28);
  if ( LOBYTE(qword_4F8D3A0[17]) )
  {
    v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    v8 = *(_QWORD **)(a2 + 112);
    v11[0] = (__int64)&unk_4F8144C;
    if ( v4 == sub_29738E0(v8, (__int64)v4, v11) )
    {
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
      {
        sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v9 + 1, 8u, v9, (__int64)v3);
        v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
      }
      *v4 = &unk_4F8144C;
      v3 = *(_QWORD **)(a2 + 112);
      v10 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
      *(_DWORD *)(a2 + 120) = v10;
      v4 = &v3[v10];
    }
  }
  else
  {
    v3 = *(_QWORD **)(a2 + 112);
    v4 = &v3[*(unsigned int *)(a2 + 120)];
  }
  v11[0] = (__int64)&unk_4F86B74;
  if ( v4 == sub_29738E0(v3, (__int64)v4, v11) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F86B74;
    ++*(_DWORD *)(a2 + 120);
  }
  return sub_BB9660(a2, (__int64)&unk_4F8FC84);
}
