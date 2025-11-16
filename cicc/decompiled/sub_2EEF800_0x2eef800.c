// Function: sub_2EEF800
// Address: 0x2eef800
//
__int64 __fastcall sub_2EEF800(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  void *v5; // rdx
  _QWORD *v6; // rax
  _BYTE *v7; // rax
  __int64 result; // rax
  __int64 v9[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v10)(unsigned __int64 *, const __m128i **, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v11)(__int64 *, __int64); // [rsp+18h] [rbp-28h]

  v3 = a1;
  v5 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 0xEu )
  {
    v3 = sub_CB6200(a1, "- lanemask:    ", 0xFu);
  }
  else
  {
    qmemcpy(v5, "- lanemask:    ", 15);
    *(_QWORD *)(a1 + 32) += 15LL;
  }
  v6 = (_QWORD *)sub_22077B0(0x10u);
  if ( v6 )
  {
    *v6 = a2;
    v6[1] = a3;
  }
  v9[0] = (__int64)v6;
  v10 = sub_2E09350;
  v11 = sub_2E092F0;
  sub_2E092F0(v9, v3);
  v7 = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)v7 >= *(_QWORD *)(v3 + 24) )
  {
    sub_CB5D20(v3, 10);
  }
  else
  {
    *(_QWORD *)(v3 + 32) = v7 + 1;
    *v7 = 10;
  }
  result = (__int64)v10;
  if ( v10 )
    return v10((unsigned __int64 *)v9, (const __m128i **)v9, 3);
  return result;
}
