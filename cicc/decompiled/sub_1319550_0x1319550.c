// Function: sub_1319550
// Address: 0x1319550
//
__int64 __fastcall sub_1319550(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // [rsp+28h] [rbp-68h] BYREF
  __int64 v5; // [rsp+30h] [rbp-60h] BYREF
  __int64 v6; // [rsp+38h] [rbp-58h] BYREF
  struct timeval tv; // [rsp+40h] [rbp-50h] BYREF
  struct timespec abstime; // [rsp+50h] [rbp-40h] BYREF

  ++*(_QWORD *)(a1 + 192);
  *(_QWORD *)(a1 + 184) = 0;
  gettimeofday(&tv, 0);
  sub_130B0D0(&v4, tv.tv_sec, 1000 * tv.tv_usec);
  if ( a2 == -1 )
  {
    *(_BYTE *)(a1 + 172) = 1;
    sub_130B0C0((_QWORD *)(a1 + 176), -1);
    pthread_cond_wait((pthread_cond_t *)(a1 + 8), (pthread_mutex_t *)(a1 + 120));
  }
  else
  {
    sub_130B270(&v5);
    sub_130B1E0(&v5, a2);
    v2 = sub_130B0E0((__int64)&v5);
    *(_BYTE *)(a1 + 172) = v2 == -1;
    sub_130B0C0((_QWORD *)(a1 + 176), v2);
    sub_130B140(&v6, &v4);
    sub_130B1E0(&v6, a2);
    abstime.tv_sec = sub_130B0F0(&v6);
    abstime.tv_nsec = sub_130B110(&v6);
    pthread_cond_timedwait((pthread_cond_t *)(a1 + 8), (pthread_mutex_t *)(a1 + 120), &abstime);
  }
  gettimeofday(&tv, 0);
  sub_130B0D0(&abstime, tv.tv_sec, 1000 * tv.tv_usec);
  result = sub_130B150(&abstime, &v4);
  if ( (int)result > 0 )
  {
    sub_130B1F0(&abstime, &v4);
    return sub_130B1D0((_QWORD *)(a1 + 200), &abstime.tv_sec);
  }
  return result;
}
