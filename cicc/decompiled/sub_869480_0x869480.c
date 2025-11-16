// Function: sub_869480
// Address: 0x869480
//
__int64 __fastcall sub_869480(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 v6; // rdx
  __int64 v7; // rdx
  const __m128i *v8; // rbx
  __int64 result; // rax
  _QWORD *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // [rsp-8h] [rbp-38h]
  int v13[9]; // [rsp+Ch] [rbp-24h] BYREF

  v6 = *a3;
  if ( !v6 )
  {
    v11 = sub_8A3C00(a2, 0, 0, 0);
    *a3 = v11;
    v6 = v11;
  }
  v13[0] = 0;
  v8 = sub_8680C0(a1, a2, v6, 0, 1, 0, 0, v13);
  result = 0;
  if ( v8 )
  {
    result = sub_85B260(a1, v12, v7);
    *(_QWORD *)(result + 8) = a1;
    v10 = qword_4F04C18;
    *(_QWORD *)(result + 16) = v8;
    *(_DWORD *)(result + 48) = 0;
    *(_QWORD *)result = v10;
    qword_4F04C18 = (_QWORD *)result;
    *(_WORD *)(result + 40) = 256;
  }
  *a4 = result;
  return result;
}
