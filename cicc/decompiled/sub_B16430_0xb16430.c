// Function: sub_B16430
// Address: 0xb16430
//
void __fastcall sub_B16430(__int64 a1, _BYTE *a2, size_t a3, _BYTE *a4, __int64 a5)
{
  size_t v5; // rax
  void *v10; // rdi
  __int64 v11; // rax
  size_t v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = a3;
  v10 = (void *)(a1 + 16);
  *(_QWORD *)a1 = v10;
  if ( &a2[a3] && !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v12[0] = a3;
  if ( a3 > 0xF )
  {
    v11 = sub_22409D0(a1, v12, 0);
    *(_QWORD *)a1 = v11;
    v10 = (void *)v11;
    *(_QWORD *)(a1 + 16) = v12[0];
LABEL_13:
    memcpy(v10, a2, a3);
    v5 = v12[0];
    v10 = *(void **)a1;
    goto LABEL_6;
  }
  if ( a3 == 1 )
  {
    *(_BYTE *)(a1 + 16) = *a2;
    goto LABEL_6;
  }
  if ( a3 )
    goto LABEL_13;
LABEL_6:
  *(_QWORD *)(a1 + 8) = v5;
  *((_BYTE *)v10 + v5) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  if ( a4 )
  {
    sub_B14B30((__int64 *)(a1 + 32), a4, (__int64)&a4[a5]);
  }
  else
  {
    *(_QWORD *)(a1 + 40) = 0;
    *(_BYTE *)(a1 + 48) = 0;
  }
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
}
