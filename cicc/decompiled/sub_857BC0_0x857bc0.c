// Function: sub_857BC0
// Address: 0x857bc0
//
__int64 __fastcall sub_857BC0(_QWORD *a1, unsigned int *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 result; // rax
  __m128i *v8; // rax
  unsigned __int64 v9; // rdi
  __m128i *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23[9]; // [rsp+Ch] [rbp-24h] BYREF

  sub_7C9660((__int64)a1);
  if ( word_4F06418[0] != 7 )
    goto LABEL_2;
  if ( !unk_4F063AD )
    return sub_7C96B0(1u, a2, v3, v4, v5, v6);
  if ( (unk_4F063A8 & 7) != 0 )
  {
LABEL_2:
    a2 = dword_4F07508;
    sub_6851C0(0x297u, dword_4F07508);
    return sub_7C96B0(1u, a2, v3, v4, v5, v6);
  }
  sub_7296C0(v23);
  v8 = sub_740630(xmmword_4F06300);
  v9 = v23[0];
  v10 = v8;
  sub_729730(v23[0]);
  sub_7B8B50(v9, a2, v11, v12, v13, v14);
  if ( word_4F06418[0] == 9 )
  {
    sub_7C96B0(0, a2, v15, v16, v17, v18);
  }
  else
  {
    sub_684B30(0xEu, dword_4F07508);
    sub_7C96B0(1u, dword_4F07508, v19, v20, v21, v22);
  }
  sub_8543B0(a1, 0, 0);
  result = a1[11];
  if ( result )
    *(_QWORD *)(result + 56) = v10;
  return result;
}
