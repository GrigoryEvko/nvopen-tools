// Function: sub_1B46A50
// Address: 0x1b46a50
//
__int64 __fastcall sub_1B46A50(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  bool v17; // zf
  __int64 v18; // rax
  _BYTE *v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // r9d
  unsigned int v23; // r8d
  int v24; // edx
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+10h] [rbp-90h]
  unsigned int v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  unsigned __int8 v32; // [rsp+18h] [rbp-88h]
  _BYTE *v33; // [rsp+20h] [rbp-80h] BYREF
  __int64 v34; // [rsp+28h] [rbp-78h]
  _BYTE v35[112]; // [rsp+30h] [rbp-70h] BYREF

  result = 0;
  v3 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v3 + 16) == 13 )
  {
    v4 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v4 + 16) == 13 )
    {
      v31 = *(_QWORD *)(a2 - 72);
      v6 = ((*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) >> 1) - 1;
      v7 = sub_1B44DF0(a1, 0, a1, v6, *(_QWORD *)(a2 - 48));
      if ( v6 == v8 )
      {
        v7 = a1;
        v9 = 24;
      }
      else if ( (_DWORD)v8 == -2 )
      {
        v9 = 24;
      }
      else
      {
        v9 = 24LL * (unsigned int)(2 * v8 + 3);
      }
      if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
        v10 = *(_QWORD *)(v7 - 8);
      else
        v10 = v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
      v11 = *(_QWORD *)(v10 + v9);
      v12 = sub_1B44DF0(a1, 0, a1, v6, v4);
      if ( v6 == v13 )
      {
        v12 = a1;
        v14 = 24;
      }
      else if ( (_DWORD)v13 == -2 )
      {
        v14 = 24;
      }
      else
      {
        v14 = 24LL * (unsigned int)(2 * v13 + 3);
      }
      if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
        v15 = *(_QWORD *)(v12 - 8);
      else
        v15 = v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
      v16 = *(_QWORD *)(v15 + v14);
      v17 = *(_QWORD *)(a1 + 48) == 0;
      v33 = v35;
      v34 = 0x800000000LL;
      if ( v17 && *(__int16 *)(a1 + 18) >= 0 )
        goto LABEL_19;
      v29 = v16;
      v18 = sub_1625790(a1, 2);
      v16 = v29;
      if ( !v18 )
        goto LABEL_19;
      v19 = *(_BYTE **)(v18 - 8LL * *(unsigned int *)(v18 + 8));
      if ( v19
        && !*v19
        && (v20 = sub_161E970((__int64)v19), v16 = v29, v21 == 14)
        && *(_QWORD *)v20 == 0x775F68636E617262LL
        && *(_DWORD *)(v20 + 8) == 1751607653
        && *(_WORD *)(v20 + 12) == 29556
        && (sub_1B43970(a1, (__int64)&v33), v16 = v29, (*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) >> 1 == (_DWORD)v34) )
      {
        sub_1B46A00(a1, v3);
        v25 = 0;
        if ( v24 != -2 )
          v25 = 8LL * (unsigned int)(v24 + 1);
        v28 = v29;
        v30 = *(_DWORD *)&v33[v25];
        sub_1B46A00(a1, v4);
        v27 = 0;
        v23 = v30;
        v16 = v28;
        if ( v26 != -2 )
          v27 = 8LL * (unsigned int)(v26 + 1);
        v22 = *(_DWORD *)&v33[v27];
      }
      else
      {
LABEL_19:
        v22 = 0;
        v23 = 0;
      }
      result = sub_1B45090(a1, v31, v11, v16, v23, v22);
      if ( v33 != v35 )
      {
        v32 = result;
        _libc_free((unsigned __int64)v33);
        return v32;
      }
    }
  }
  return result;
}
