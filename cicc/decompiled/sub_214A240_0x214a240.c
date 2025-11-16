// Function: sub_214A240
// Address: 0x214a240
//
__int64 __fastcall sub_214A240(__int64 a1, __int64 a2, __int64 a3)
{
  char *v3; // r15
  unsigned __int8 *v4; // r12
  unsigned __int8 *v5; // rbx
  _QWORD *v6; // r14
  size_t v7; // rax
  __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  __int64 result; // rax
  unsigned __int8 *v11; // [rsp+0h] [rbp-180h]
  unsigned int v13; // [rsp+40h] [rbp-140h]
  unsigned int v14; // [rsp+48h] [rbp-138h]
  int v15; // [rsp+4Ch] [rbp-134h]
  size_t v16; // [rsp+50h] [rbp-130h]
  char *src; // [rsp+58h] [rbp-128h]
  _QWORD v18[2]; // [rsp+60h] [rbp-120h] BYREF
  _QWORD *v19; // [rsp+70h] [rbp-110h] BYREF
  __int16 v20; // [rsp+80h] [rbp-100h]
  _QWORD v21[2]; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v22; // [rsp+A0h] [rbp-E0h]
  void *dest; // [rsp+A8h] [rbp-D8h]
  int v24; // [rsp+B0h] [rbp-D0h]
  unsigned __int64 *v25; // [rsp+B8h] [rbp-C8h]
  unsigned __int64 v26[2]; // [rsp+C0h] [rbp-C0h] BYREF
  _BYTE v27[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v14 = 40;
  v15 = 0;
  src = *(char **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 16LL) + 200LL);
  v11 = (unsigned __int8 *)(a2 + a3);
  v13 = ((int)a3 - 1) / 0x28u;
  while ( 1 )
  {
    v24 = 1;
    v26[0] = (unsigned __int64)v27;
    v26[1] = 0x8000000000LL;
    dest = 0;
    v21[0] = &unk_49EFC48;
    v22 = 0;
    v21[1] = 0;
    v25 = v26;
    sub_16E7A40((__int64)v21, 0, 0, 0);
    v3 = src;
    v4 = (unsigned __int8 *)(a2 + v14);
    v5 = (unsigned __int8 *)(a2 + v14 - 40);
    if ( v13 == v15 )
      v4 = v11;
    for ( ; v4 != v5; ++v5 )
    {
      v6 = v21;
      if ( v3 )
      {
        v7 = strlen(v3);
        if ( v7 <= v22 - (__int64)dest )
        {
          if ( v7 )
          {
            v16 = v7;
            memcpy(dest, v3, v7);
            dest = (char *)dest + v16;
          }
        }
        else
        {
          v6 = (_QWORD *)sub_16E7EE0((__int64)v21, v3, v7);
        }
      }
      sub_16E7A90((__int64)v6, *v5);
      if ( v3 == src )
        v3 = ",";
    }
    v8 = *(_QWORD *)(a1 + 8);
    v9 = *v25;
    v18[1] = *((unsigned int *)v25 + 2);
    v20 = 261;
    v18[0] = v9;
    v19 = v18;
    sub_38DD5A0(v8, &v19);
    v21[0] = &unk_49EFD28;
    sub_16E7960((__int64)v21);
    if ( (_BYTE *)v26[0] != v27 )
      _libc_free(v26[0]);
    v14 += 40;
    result = (unsigned int)(v15 + 1);
    if ( v13 == v15 )
      break;
    ++v15;
  }
  return result;
}
