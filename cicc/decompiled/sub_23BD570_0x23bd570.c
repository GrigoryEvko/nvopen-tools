// Function: sub_23BD570
// Address: 0x23bd570
//
__int64 __fastcall sub_23BD570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  void *v12; // rdx
  __int64 result; // rax
  _QWORD v14[3]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v15; // [rsp+18h] [rbp-48h] BYREF
  __int64 v16[2]; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v17[2]; // [rsp+30h] [rbp-30h] BYREF
  __int64 (__fastcall *v18)(unsigned __int64 *, const __m128i **, int); // [rsp+40h] [rbp-20h]
  __int64 (__fastcall *v19)(); // [rsp+48h] [rbp-18h]

  v14[0] = a2;
  v14[1] = a3;
  v16[0] = a5;
  v16[1] = a6;
  v18 = 0;
  v9 = (_QWORD *)sub_22077B0(0x18u);
  if ( v9 )
  {
    *v9 = a4;
    v9[1] = v14;
    v9[2] = a1;
  }
  v17[0] = v9;
  v19 = sub_23C61D0;
  v18 = sub_23AE810;
  sub_23B2720(&v15, a7);
  v10 = sub_23B27D0(&v15);
  sub_23BD210(v16, v10 != 0, (__int64)v17);
  if ( v15 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 8LL))(v15);
  if ( v18 )
    v18(v17, (const __m128i **)v17, 3);
  v11 = *(_QWORD *)(a1 + 40);
  v12 = *(void **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 0xEu )
  {
    result = sub_CB6200(v11, "    </p></div>\n", 0xFu);
  }
  else
  {
    qmemcpy(v12, "    </p></div>\n", 15);
    result = 15990;
    *(_QWORD *)(v11 + 32) += 15LL;
  }
  ++*(_DWORD *)(a1 + 36);
  return result;
}
