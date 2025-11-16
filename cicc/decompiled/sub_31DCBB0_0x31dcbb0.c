// Function: sub_31DCBB0
// Address: 0x31dcbb0
//
void __fastcall sub_31DCBB0(__int64 a1, int a2)
{
  __int64 (*v2)(); // rax
  int v3; // ebx
  __int64 v4; // rax
  _BYTE v5[16]; // [rsp+0h] [rbp-A0h] BYREF
  char *v6; // [rsp+10h] [rbp-90h]
  char v7; // [rsp+20h] [rbp-80h] BYREF

  v2 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 232) + 16LL) + 128LL);
  if ( v2 == sub_2DAC790 )
    BUG();
  v3 = a2;
  v4 = v2();
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 904LL))(v5, v4);
  if ( a2 )
  {
    do
    {
      sub_31DB460(a1, *(_QWORD *)(a1 + 224), (__int64)v5);
      --v3;
    }
    while ( v3 );
  }
  if ( v6 != &v7 )
    _libc_free((unsigned __int64)v6);
}
