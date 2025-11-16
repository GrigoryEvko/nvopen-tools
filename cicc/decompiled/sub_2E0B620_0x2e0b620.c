// Function: sub_2E0B620
// Address: 0x2e0b620
//
__int64 __fastcall sub_2E0B620(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // rbx
  _QWORD *v6; // rax
  _BYTE *v7; // rax
  __int64 result; // rax
  __int64 v9[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v10)(unsigned __int64 *, const __m128i **, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v11)(__int64 *, __int64); // [rsp+18h] [rbp-28h]

  v2 = a2;
  v3 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v3) <= 2 )
  {
    v2 = sub_CB6200(a2, (unsigned __int8 *)&unk_444FD20, 3u);
  }
  else
  {
    *(_BYTE *)(v3 + 2) = 76;
    *(_WORD *)v3 = 8224;
    *(_QWORD *)(a2 + 32) += 3LL;
  }
  v4 = *(_QWORD *)(a1 + 112);
  v5 = *(_QWORD *)(a1 + 120);
  v6 = (_QWORD *)sub_22077B0(0x10u);
  if ( v6 )
  {
    *v6 = v4;
    v6[1] = v5;
  }
  v9[0] = (__int64)v6;
  v10 = sub_2E09350;
  v11 = sub_2E092F0;
  sub_2E092F0(v9, v2);
  v7 = *(_BYTE **)(v2 + 32);
  if ( (unsigned __int64)v7 >= *(_QWORD *)(v2 + 24) )
  {
    v2 = sub_CB5D20(v2, 32);
  }
  else
  {
    *(_QWORD *)(v2 + 32) = v7 + 1;
    *v7 = 32;
  }
  sub_2E0B3F0(a1, v2);
  result = (__int64)v10;
  if ( v10 )
    return v10((unsigned __int64 *)v9, (const __m128i **)v9, 3);
  return result;
}
