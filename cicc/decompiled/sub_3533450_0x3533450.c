// Function: sub_3533450
// Address: 0x3533450
//
void __fastcall sub_3533450(char *a1, char *a2)
{
  __int64 (__fastcall ***v2)(_QWORD); // r13
  char *v3; // r15
  _QWORD *v4; // r14
  int v5; // eax
  unsigned int v6; // r12d
  __int64 v7; // r13
  int v8; // eax
  int v9; // ebx
  int v10; // eax
  _DWORD *v11; // r12
  _QWORD *v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdi
  _DWORD *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdi
  int v20; // eax
  int v21; // r13d
  unsigned int v22; // r12d
  __int64 v23; // r13
  int v24; // eax
  char *v25; // [rsp+8h] [rbp-48h]
  char *v26; // [rsp+10h] [rbp-40h]

  if ( a1 != a2 && a1 + 8 != a2 )
  {
    v26 = a1 + 8;
    do
    {
      v2 = *(__int64 (__fastcall ****)(_QWORD))v26;
      v25 = v26;
      v3 = v26;
      v4 = v26 + 8;
      v5 = (***(__int64 (__fastcall ****)(_QWORD))v26)(*(_QWORD *)v26);
      LODWORD(v2) = *((_DWORD *)v2 + 10);
      v6 = (_DWORD)v2 * (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)a1 + 8LL))(*(_QWORD *)a1) * v5;
      v7 = *(_QWORD *)a1;
      v8 = (***(__int64 (__fastcall ****)(_QWORD))a1)(*(_QWORD *)a1);
      LODWORD(v7) = *(_DWORD *)(v7 + 40);
      v9 = v8;
      v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)v26 + 8LL))(*(_QWORD *)v26);
      v26 += 8;
      if ( v6 <= v10 * v9 * (int)v7 )
      {
        v17 = *(_DWORD **)v25;
        *(_QWORD *)v25 = 0;
        while ( 1 )
        {
          v20 = (**(__int64 (__fastcall ***)(_DWORD *, char *))v17)(v17, v25);
          v21 = v17[10];
          v22 = v21 * (*(__int64 (__fastcall **)(_QWORD))(**((_QWORD **)v3 - 1) + 8LL))(*((_QWORD *)v3 - 1)) * v20;
          v23 = *((_QWORD *)v3 - 1);
          v24 = (**(__int64 (__fastcall ***)(__int64))v23)(v23);
          LODWORD(v23) = *(_DWORD *)(v23 + 40);
          if ( v22 <= (*(unsigned int (__fastcall **)(_DWORD *))(*(_QWORD *)v17 + 8LL))(v17) * v24 * (unsigned int)v23 )
            break;
          v18 = *((_QWORD *)v3 - 1);
          v19 = *(_QWORD *)v3;
          *((_QWORD *)v3 - 1) = 0;
          *(_QWORD *)v3 = v18;
          if ( v19 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 24LL))(v19);
          v3 -= 8;
        }
        v16 = *(_QWORD *)v3;
        *(_QWORD *)v3 = v17;
        if ( !v16 )
          continue;
      }
      else
      {
        v11 = *(_DWORD **)v25;
        *(_QWORD *)v25 = 0;
        v12 = v4;
        v13 = (v25 - a1) >> 3;
        if ( v25 - a1 > 0 )
        {
          do
          {
            v14 = *(v12 - 2);
            v15 = *--v12;
            *(v12 - 1) = 0;
            *v12 = v14;
            if ( v15 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 24LL))(v15);
            --v13;
          }
          while ( v13 );
        }
        v16 = *(_QWORD *)a1;
        *(_QWORD *)a1 = v11;
        if ( !v16 )
          continue;
      }
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 24LL))(v16);
    }
    while ( a2 != v26 );
  }
}
