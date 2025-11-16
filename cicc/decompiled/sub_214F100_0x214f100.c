// Function: sub_214F100
// Address: 0x214f100
//
_QWORD *__fastcall sub_214F100(__int64 a1, unsigned int a2, __int64 a3)
{
  _QWORD *result; // rax
  char *v5[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v6[3]; // [rsp+10h] [rbp-20h] BYREF

  (*(void (__fastcall **)(char **, __int64, _QWORD))(*(_QWORD *)a1 + 392LL))(v5, a1, a2);
  sub_16E7EE0(a3, v5[0], (size_t)v5[1]);
  result = v6;
  if ( (_QWORD *)v5[0] != v6 )
    return (_QWORD *)j_j___libc_free_0(v5[0], v6[0] + 1LL);
  return result;
}
