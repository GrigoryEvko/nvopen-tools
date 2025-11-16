// Function: sub_324A9A0
// Address: 0x324a9a0
//
void __fastcall sub_324A9A0(__int64 **a1, __int16 a2, __int64 a3)
{
  unsigned __int8 *v3; // rcx
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 *v8; // r14
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned __int64 *v11[2]; // [rsp-D8h] [rbp-D8h] BYREF
  _QWORD v12[3]; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 *v13; // [rsp-B0h] [rbp-B0h]
  __int64 v14; // [rsp-A0h] [rbp-A0h] BYREF
  char v15; // [rsp-64h] [rbp-64h]
  __int64 **v16; // [rsp-58h] [rbp-58h]

  if ( a3 )
  {
    if ( ((a3 >> 2) & 1) != 0 )
    {
      if ( ((a3 >> 2) & 1) != 0 )
      {
        v4 = a3 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (a3 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v11[0] = (unsigned __int64 *)sub_AF4F20(a3 & 0xFFFFFFFFFFFFFFF8LL);
          if ( BYTE4(v11[0]) && (v12[0] = sub_AF4F20(v4), !LODWORD(v12[0])) )
          {
            v9 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL);
            if ( a2 != 34 || (v10 = *a1[2], v10 == -1) || v10 != v9 )
              sub_32498F0(*a1, (unsigned __int64 **)a1[1] + 1, a2, 65549, v9);
          }
          else
          {
            v5 = sub_A777F0(0x10u, *a1 + 11);
            v6 = v5;
            if ( v5 )
            {
              *(_QWORD *)v5 = 0;
              *(_DWORD *)(v5 + 8) = 0;
            }
            v7 = (*(__int64 (__fastcall **)(__int64 *))(**a1 + 72))(*a1);
            sub_3247620((__int64)v12, (*a1)[23], v7, v6);
            v15 = v15 & 0xF8 | 2;
            v11[0] = *(unsigned __int64 **)(v4 + 16);
            v11[1] = *(unsigned __int64 **)(v4 + 24);
            sub_3244870(v12, v11);
            v8 = *a1;
            sub_3243D40((__int64)v12);
            sub_3249620(v8, (__int64)a1[1], a2, v16);
            if ( v13 != &v14 )
              _libc_free((unsigned __int64)v13);
          }
        }
      }
    }
    else if ( (a3 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v3 = sub_3247C80((__int64)*a1, (unsigned __int8 *)(a3 & 0xFFFFFFFFFFFFFFF8LL));
      if ( v3 )
        sub_32494F0(*a1, (unsigned __int64)a1[1], a2, (unsigned __int64)v3);
    }
  }
}
