// Function: sub_7AC370
// Address: 0x7ac370
//
void *__fastcall sub_7AC370(void *src, size_t n)
{
  __int64 v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // rax
  _QWORD *v7; // rax
  size_t v8; // rdi
  __int64 *v9; // rdx

  v3 = qword_4F084B0;
  if ( qword_4F084B0 && (v4 = *(_QWORD *)(qword_4F084B0 + 16), *(_QWORD *)(qword_4F084B0 + 8) - v4 >= n) )
  {
    v5 = *(_QWORD *)(qword_4F084B0 + 24);
  }
  else
  {
    v7 = (_QWORD *)sub_823970(32);
    v8 = 65000;
    if ( n >= 0xFDE8 )
      v8 = n;
    *v7 = 0;
    v3 = (__int64)v7;
    v7[2] = 0;
    v7[1] = v8;
    v5 = sub_823970(v8);
    v9 = (__int64 *)qword_4F084B0;
    *(_QWORD *)(v3 + 24) = v5;
    if ( v9 )
    {
      *v9 = v3;
      v5 = *(_QWORD *)(v3 + 24);
    }
    else
    {
      qword_4F084B8 = v3;
    }
    qword_4F084B0 = v3;
    v4 = *(_QWORD *)(v3 + 16);
  }
  *(_QWORD *)(v3 + 16) = n + v4;
  return memcpy((void *)(v5 + v4), src, n);
}
