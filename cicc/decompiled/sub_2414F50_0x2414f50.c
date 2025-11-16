// Function: sub_2414F50
// Address: 0x2414f50
//
__int64 __fastcall sub_2414F50(__int64 a1, __int64 a2, char *a3, __int64 a4, __int16 a5)
{
  _QWORD *v9; // r13
  char v10; // dl
  __int64 v11; // rax
  unsigned int v12; // r8d
  __int64 v13; // rcx
  __int64 v14; // rdi
  int v15; // r15d
  __int64 *v16; // r10
  unsigned int v17; // esi
  __int64 *v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // r12
  bool v23; // al
  __int64 **v24; // rax
  int v25; // eax
  int v26; // edx
  int v27; // esi
  int v28; // [rsp+8h] [rbp-F8h]
  __int64 v29; // [rsp+10h] [rbp-F0h] BYREF
  __int64 *v30; // [rsp+18h] [rbp-E8h] BYREF
  unsigned __int64 v31[2]; // [rsp+20h] [rbp-E0h] BYREF
  _BYTE v32[16]; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int64 v33[2]; // [rsp+40h] [rbp-C0h] BYREF
  char v34; // [rsp+50h] [rbp-B0h] BYREF
  void *v35; // [rsp+C0h] [rbp-40h]

  v9 = sub_240F000(*(_QWORD *)a1, a2);
  if ( (unsigned __int8)(*((_BYTE *)v9 + 8) - 15) > 1u )
    return (__int64)a3;
  v10 = *a3;
  if ( (unsigned __int8)(*(_BYTE *)(*((_QWORD *)a3 + 1) + 8LL) - 15) > 1u )
  {
    if ( v10 != 17
      || (*((_DWORD *)a3 + 8) <= 0x40u
        ? (v23 = *((_QWORD *)a3 + 3) == 0)
        : (v28 = *((_DWORD *)a3 + 8), v23 = v28 == (unsigned int)sub_C444A0((__int64)(a3 + 24))),
          !v23) )
    {
LABEL_4:
      if ( !a4 )
        BUG();
      sub_2412230((__int64)v33, *(_QWORD *)(a4 + 16), a4, a5, 0, (__int64)v33, 0, 0);
      v31[0] = (unsigned __int64)v32;
      v31[1] = 0x400000000LL;
      v29 = sub_ACA8A0((__int64 **)v9);
      v11 = sub_240F6F0(v29, (__int64)v31, (__int64)v9, a3, (__int64)v33);
      v12 = *(_DWORD *)(a1 + 440);
      v29 = v11;
      v13 = v11;
      if ( v12 )
      {
        v14 = *(_QWORD *)(a1 + 424);
        v15 = 1;
        v16 = 0;
        v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v18 = (__int64 *)(v14 + 16LL * v17);
        v19 = *v18;
        if ( v13 == *v18 )
        {
LABEL_7:
          v20 = v18 + 1;
LABEL_8:
          *v20 = a3;
          v21 = v29;
          if ( (_BYTE *)v31[0] != v32 )
            _libc_free(v31[0]);
          nullsub_61();
          v35 = &unk_49DA100;
          nullsub_63();
          if ( (char *)v33[0] != &v34 )
            _libc_free(v33[0]);
          return v21;
        }
        while ( v19 != -4096 )
        {
          if ( !v16 && v19 == -8192 )
            v16 = v18;
          v17 = (v12 - 1) & (v15 + v17);
          v18 = (__int64 *)(v14 + 16LL * v17);
          v19 = *v18;
          if ( v13 == *v18 )
            goto LABEL_7;
          ++v15;
        }
        if ( !v16 )
          v16 = v18;
        v25 = *(_DWORD *)(a1 + 432);
        ++*(_QWORD *)(a1 + 416);
        v26 = v25 + 1;
        v30 = v16;
        if ( 4 * (v25 + 1) < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a1 + 436) - v26 > v12 >> 3 )
          {
LABEL_30:
            *(_DWORD *)(a1 + 432) = v26;
            if ( *v16 != -4096 )
              --*(_DWORD *)(a1 + 436);
            *v16 = v13;
            v20 = v16 + 1;
            v16[1] = 0;
            goto LABEL_8;
          }
          v27 = v12;
LABEL_35:
          sub_FAA400(a1 + 416, v27);
          sub_F9D990(a1 + 416, &v29, &v30);
          v13 = v29;
          v16 = v30;
          v26 = *(_DWORD *)(a1 + 432) + 1;
          goto LABEL_30;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 416);
        v30 = 0;
      }
      v27 = 2 * v12;
      goto LABEL_35;
    }
  }
  else if ( v10 != 14 )
  {
    goto LABEL_4;
  }
  v24 = (__int64 **)sub_240F000(*(_QWORD *)a1, (__int64)v9);
  return sub_AC9350(v24);
}
