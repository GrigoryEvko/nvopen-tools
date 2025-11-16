// Function: sub_30979D0
// Address: 0x30979d0
//
__int64 __fastcall sub_30979D0(__int64 a1, const void *a2, size_t a3)
{
  unsigned int *v3; // r13
  __int64 v4; // r8
  char *v5; // r14
  __int64 v6; // r8
  __int64 v7; // r9
  size_t v8; // r11
  __int64 v9; // rax
  char **v10; // rax
  int v11; // eax
  size_t v13; // [rsp+10h] [rbp-58h]

  v3 = (unsigned int *)&unk_44CAF00;
  sub_22F5730(a1, (__int64)&off_49D87F0, (__int64)dword_44CAF10, 4, (__int64)&unk_49D6D60, 85, 0);
  v4 = 1;
  *(_QWORD *)a1 = &unk_4A082F0;
  while ( 1 )
  {
    v5 = &byte_44CAF20[v4];
    v8 = strlen(&byte_44CAF20[v4]);
    v9 = *(unsigned int *)(a1 + 88);
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
    {
      v13 = v8;
      sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v9 + 1, 0x10u, v6, v7);
      v9 = *(unsigned int *)(a1 + 88);
      v8 = v13;
    }
    ++v3;
    v10 = (char **)(*(_QWORD *)(a1 + 80) + 16 * v9);
    *v10 = v5;
    v10[1] = (char *)v8;
    ++*(_DWORD *)(a1 + 88);
    if ( &unk_44CAF08 == (_UNKNOWN *)v3 )
      break;
    v4 = *v3;
  }
  sub_22F5820(a1);
  *(_QWORD *)a1 = &unk_4A31DB8;
  v11 = sub_30978E0(a1, a2, a3);
  *(_QWORD *)(a1 + 336) = 0;
  *(_DWORD *)(a1 + 176) = v11;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  *(_QWORD *)(a1 + 200) = 0x1000000000LL;
  *(_QWORD *)(a1 + 184) = &unk_4A0AA50;
  *(_QWORD *)(a1 + 376) = 0x1000000000LL;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_DWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = a1 + 384;
  *(_QWORD *)(a1 + 520) = a1 + 512;
  *(_QWORD *)(a1 + 512) = a1 + 512;
  *(_QWORD *)(a1 + 528) = 0;
  *(_DWORD *)(a1 + 536) = 0;
  return a1 + 512;
}
