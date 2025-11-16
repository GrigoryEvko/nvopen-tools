// Function: sub_2F60A40
// Address: 0x2f60a40
//
__int64 __fastcall sub_2F60A40(__int64 a1, unsigned int a2)
{
  _QWORD *v3; // rdi
  unsigned __int16 ****v4; // r14
  unsigned __int16 ***v5; // r13
  unsigned __int16 ***v6; // rbx
  int *v7; // rax
  int v8; // edx
  unsigned int v9; // eax
  _DWORD *v10; // rbx
  unsigned int v11; // r14d
  int v12; // ebx
  unsigned int v14; // [rsp+4h] [rbp-3Ch]
  unsigned __int16 ****v15; // [rsp+8h] [rbp-38h]
  int v16; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD **)(a1 + 24);
  v15 = (unsigned __int16 ****)v3[36];
  if ( v15 == (unsigned __int16 ****)v3[35] )
  {
    sub_2F60630(a1, 0);
    BUG();
  }
  v14 = 0;
  v4 = (unsigned __int16 ****)v3[35];
  v5 = 0;
  while ( 1 )
  {
    v6 = *v4;
    v7 = (int *)(*(__int64 (__fastcall **)(_QWORD *, unsigned __int16 ***))(*v3 + 416LL))(v3, *v4);
    v8 = *v7;
    if ( *v7 != -1 )
    {
      while ( v8 != a2 )
      {
        v8 = v7[1];
        ++v7;
        if ( v8 == -1 )
          goto LABEL_9;
      }
      v9 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD, unsigned __int16 ***))(**(_QWORD **)(a1 + 24) + 376LL))(
                         *(_QWORD *)(a1 + 24),
                         v6)
                     + 4);
      if ( !v5 || v9 > v14 )
        break;
    }
LABEL_9:
    if ( v15 == ++v4 )
      goto LABEL_12;
LABEL_10:
    v3 = *(_QWORD **)(a1 + 24);
  }
  v14 = v9;
  v5 = v6;
  if ( v15 != ++v4 )
    goto LABEL_10;
LABEL_12:
  sub_2F60630(a1, v5);
  v10 = (_DWORD *)(*(_QWORD *)a1 + 24LL * *((unsigned __int16 *)*v5 + 12));
  if ( *(_DWORD *)(a1 + 8) != *v10 )
    sub_2F60630(a1, v5);
  v16 = v10[1];
  v11 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 24) + 408LL))(
          *(_QWORD *)(a1 + 24),
          *(_QWORD *)(a1 + 16),
          a2);
  if ( v16 )
  {
    v12 = *((unsigned __int16 *)*v5 + 10) - v16;
    v11 -= *(_DWORD *)(*(__int64 (__fastcall **)(_QWORD, unsigned __int16 ***))(**(_QWORD **)(a1 + 24) + 376LL))(
                        *(_QWORD *)(a1 + 24),
                        v5)
         * v12;
  }
  return v11;
}
