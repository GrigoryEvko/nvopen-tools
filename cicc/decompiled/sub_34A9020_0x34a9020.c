// Function: sub_34A9020
// Address: 0x34a9020
//
void __fastcall sub_34A9020(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 *v12; // rdi
  unsigned __int64 *v13; // rbx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned __int64 *v18; // rdx
  _QWORD *v19; // rdx
  __int64 *v20; // rax
  unsigned __int64 *v21; // [rsp+8h] [rbp-148h]
  unsigned __int64 v22; // [rsp+10h] [rbp-140h]
  unsigned __int64 v23; // [rsp+18h] [rbp-138h]
  unsigned __int64 v24; // [rsp+20h] [rbp-130h]
  __int64 v25; // [rsp+30h] [rbp-120h] BYREF
  _QWORD *v26; // [rsp+38h] [rbp-118h]
  __int64 v27; // [rsp+40h] [rbp-110h]
  _QWORD v28[9]; // [rsp+48h] [rbp-108h] BYREF
  unsigned __int64 *v29; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+98h] [rbp-B8h]
  _BYTE v31[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v29 = (unsigned __int64 *)v31;
  v30 = 0x800000000LL;
  v7 = sub_34A5A20(a1, a2, (__int64)&v29, a4, a5, a6);
  v12 = v29;
  if ( v7 )
  {
    v21 = &v29[2 * (unsigned int)v30];
    if ( v21 != v29 )
    {
      v13 = v29;
      do
      {
        v14 = v13[1];
        v15 = *v13;
        v25 = a1 + 8;
        v24 = v14;
        v26 = v28;
        v27 = 0x400000000LL;
        v16 = *(unsigned int *)(a1 + 200);
        if ( (_DWORD)v16 )
        {
          a2 = v15;
          sub_34A3C90((__int64)&v25, v15, v8, v9, v10, v11);
          v19 = &v26[2 * (unsigned int)v27 - 2];
        }
        else
        {
          v17 = *(unsigned int *)(a1 + 204);
          if ( (_DWORD)v17 )
          {
            v18 = (unsigned __int64 *)(a1 + 16);
            do
            {
              if ( v15 <= *v18 )
                break;
              v16 = (unsigned int)(v16 + 1);
              v18 += 2;
            }
            while ( (_DWORD)v17 != (_DWORD)v16 );
          }
          v28[0] = a1 + 8;
          v19 = v28;
          LODWORD(v27) = 1;
          v28[1] = v17 | (v16 << 32);
        }
        v20 = (__int64 *)(*v19 + 16LL * *((unsigned int *)v19 + 3));
        v22 = *v20;
        v23 = v20[1];
        sub_34A3230((__int64)&v25, a2, (__int64)v19, v17, *v20);
        v10 = v22;
        if ( v15 > v22 )
          sub_34A8ED0(a1 + 8, v22, v15 - 1, 0, v22, v11);
        a2 = v23;
        if ( v24 < v23 )
        {
          a2 = v24 + 1;
          sub_34A8ED0(a1 + 8, v24 + 1, v23, 0, v10, v11);
        }
        if ( v26 != v28 )
          _libc_free((unsigned __int64)v26);
        v13 += 2;
      }
      while ( v21 != v13 );
      v12 = v29;
    }
  }
  if ( v12 != (unsigned __int64 *)v31 )
    _libc_free((unsigned __int64)v12);
}
