// Function: sub_C65D30
// Address: 0xc65d30
//
void *__fastcall sub_C65D30(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 v4; // rdx
  __int64 v5; // rsi
  unsigned __int64 v6; // rcx
  void *v7; // r8
  size_t v8; // r9

  v4 = *a2;
  v5 = 4LL * *(unsigned int *)(a1 + 8);
  a2[10] += v5;
  v6 = v5 + ((v4 + 3) & 0xFFFFFFFFFFFFFFFCLL);
  if ( a2[1] >= v6 && v4 )
  {
    *a2 = v6;
    v7 = (void *)((v4 + 3) & 0xFFFFFFFFFFFFFFFCLL);
  }
  else
  {
    v7 = (void *)sub_9D1E70((__int64)a2, v5, v5, 2);
  }
  v8 = 4LL * *(unsigned int *)(a1 + 8);
  if ( v8 )
    return memmove(v7, *(const void **)a1, v8);
  return v7;
}
