// Function: sub_39A3F30
// Address: 0x39a3f30
//
__int64 __fastcall sub_39A3F30(__int64 *a1, __int64 a2, __int16 a3, void *a4, size_t a5)
{
  __int64 result; // rax
  __int64 *v8; // r13
  __int64 v10; // r12
  __int16 v11; // ax
  unsigned int v12; // edx
  __int64 v13; // rax
  void *v14; // rcx
  __int64 *v15; // r8
  _QWORD *v16; // rbx
  void *v17; // rax
  void *v18; // rax
  __int64 v19; // [rsp-48h] [rbp-48h] BYREF
  _QWORD *v20; // [rsp-40h] [rbp-40h]

  result = a1[10];
  if ( *(_DWORD *)(result + 36) != 3 )
  {
    v8 = (__int64 *)(a2 + 8);
    if ( *(_BYTE *)(a1[25] + 4499) )
    {
      v13 = sub_145CBF0(a1 + 11, 16, 16);
      v14 = 0;
      v15 = a1 + 11;
      v16 = (_QWORD *)v13;
      if ( a5 )
      {
        v17 = (void *)sub_145CBF0(a1 + 11, a5, 1);
        v18 = memcpy(v17, a4, a5);
        v15 = a1 + 11;
        v14 = v18;
      }
      v16[1] = a5;
      *v16 = v14;
      WORD2(v19) = a3;
      v20 = v16;
      LODWORD(v19) = 10;
      HIWORD(v19) = 8;
      return sub_39A31C0(v8, v15, &v19);
    }
    else
    {
      v10 = sub_39A1860(a1[26] + 192, a1[24], (unsigned __int8 *)a4, a5);
      v11 = (*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 56))(a1) == 0 ? 14 : 7938;
      if ( *(_BYTE *)(a1[25] + 4514) )
      {
        v12 = *(_DWORD *)(v10 + 20);
        v11 = 40;
        if ( v12 <= 0xFFFFFF )
        {
          v11 = 39;
          if ( v12 <= 0xFFFF )
            v11 = (v12 > 0xFF) + 37;
        }
      }
      WORD2(v19) = a3;
      v20 = (_QWORD *)v10;
      LODWORD(v19) = 2;
      HIWORD(v19) = v11;
      return sub_39A31C0(v8, a1 + 11, &v19);
    }
  }
  return result;
}
