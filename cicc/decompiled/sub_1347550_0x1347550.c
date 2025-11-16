// Function: sub_1347550
// Address: 0x1347550
//
void __fastcall sub_1347550(__int64 a1, __int64 a2, char a3)
{
  char v3; // bl
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r12
  char v9; // bl
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v15; // [rsp+18h] [rbp-D8h]
  __int64 v16; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v17; // [rsp+38h] [rbp-B8h]
  __int64 v18; // [rsp+40h] [rbp-B0h]
  __int64 v19; // [rsp+48h] [rbp-A8h]
  __int64 v20; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v21; // [rsp+58h] [rbp-98h] BYREF
  _QWORD v22[18]; // [rsp+60h] [rbp-90h] BYREF

  if ( a3 )
  {
    v15 = -1;
  }
  else
  {
    v15 = 16;
    if ( *(_BYTE *)(a2 + 5644) )
      return;
  }
  v17 = 0;
  v14 = a2 + 64;
  do
  {
LABEL_4:
    v3 = 0;
    while ( 1 )
    {
      v4 = *(unsigned int *)(a2 + 5640);
      if ( (_DWORD)v4 == -1
        || ((v5 = *(_QWORD *)(a2 + 1360), v5 > 0xFFFFFFFFFFFFLL) ? (v6 = v4 * (v5 >> 16)) : (v6 = (v4 * v5) >> 16),
            *(_QWORD *)(a2 + 1368) - *(_QWORD *)(a2 + 5664) <= v6) )
      {
        if ( !sub_1347290(a2) )
          break;
      }
      if ( v17 >= v15 )
        break;
      v18 = a2 + 320;
      v7 = sub_134BF60(a2 + 320);
      v8 = v7;
      if ( !v7 )
        goto LABEL_4;
      sub_134BCA0(v18, v7);
      *(_BYTE *)(v8 + 33) = 1;
      *(_WORD *)(v8 + 19) = 0;
      *(_BYTE *)(v8 + 17) = 0;
      sub_134BD00(v18, v8);
      v9 = *(_BYTE *)(v8 + 16);
      v10 = sub_1349FB0(v8, v22);
      *(_QWORD *)(a2 + 5664) += v10;
      v16 = v10;
      *(_BYTE *)(a2 + 168) = 0;
      pthread_mutex_unlock((pthread_mutex_t *)(a2 + 128));
      if ( v9 )
        (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)(a2 + 56) + 288LL))(*(_QWORD *)v8, 0x200000);
      v19 = 0;
      while ( (unsigned __int8)sub_134A330(v8, v22, &v20, &v21) )
      {
        ++v19;
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)(a2 + 56) + 272LL))(v20, v21);
      }
      if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 128)) )
      {
        sub_130AD90(v14);
        *(_BYTE *)(a2 + 168) = 1;
      }
      ++*(_QWORD *)(a2 + 120);
      if ( a1 != *(_QWORD *)(a2 + 112) )
      {
        ++*(_QWORD *)(a2 + 104);
        *(_QWORD *)(a2 + 112) = a1;
      }
      *(_QWORD *)(a2 + 5664) -= v16;
      *(_QWORD *)(a2 + 5680) += v19;
      v11 = *(_QWORD *)(a2 + 56);
      ++*(_QWORD *)(a2 + 5672);
      (*(void (__fastcall **)(__int64, _QWORD))(v11 + 296))(a2 + 5704, 0);
      if ( v9 )
      {
        ++*(_QWORD *)(a2 + 5696);
        sub_134BCA0(v18, v8);
        sub_134A510(v8);
      }
      else
      {
        sub_134BCA0(v18, v8);
      }
      v3 = 1;
      sub_134A480(v8, v22);
      *(_BYTE *)(v8 + 33) = 0;
      *(_BYTE *)(v8 + 17) = 1;
      sub_1347380(a2, v8);
      sub_134BD00(v18, v8);
      ++v17;
    }
    if ( sub_1347290(a2)
      || (v12 = sub_134BFA0(a2 + 320), (v13 = v12) == 0)
      || (v22[0] = *(_QWORD *)(v12 + 24),
          (unsigned __int64)(*(__int64 (__fastcall **)(_QWORD *))(*(_QWORD *)(a2 + 56) + 304LL))(v22) < *(_QWORD *)(a2 + 5648)) )
    {
      if ( !v3 )
        return;
    }
    else
    {
      sub_134BCA0(a2 + 320, v13);
      *(_BYTE *)(v13 + 34) = 1;
      *(_WORD *)(v13 + 19) = 0;
      sub_134BD00(a2 + 320, v13);
      *(_BYTE *)(a2 + 168) = 0;
      pthread_mutex_unlock((pthread_mutex_t *)(a2 + 128));
      (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)(a2 + 56) + 280LL))(*(_QWORD *)v13, 0x200000);
      if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 128)) )
      {
        sub_130AD90(v14);
        *(_BYTE *)(a2 + 168) = 1;
      }
      ++*(_QWORD *)(a2 + 120);
      if ( a1 != *(_QWORD *)(a2 + 112) )
      {
        ++*(_QWORD *)(a2 + 104);
        *(_QWORD *)(a2 + 112) = a1;
      }
      ++*(_QWORD *)(a2 + 5688);
      sub_134BCA0(a2 + 320, v13);
      sub_134A4D0(v13);
      *(_BYTE *)(v13 + 34) = 0;
      sub_1347380(a2, v13);
      sub_134BD00(a2 + 320, v13);
      ++v17;
    }
  }
  while ( v15 > v17 );
}
