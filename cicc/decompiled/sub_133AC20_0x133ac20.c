// Function: sub_133AC20
// Address: 0x133ac20
//
__int64 __fastcall sub_133AC20(_BYTE *a1)
{
  unsigned int v2; // r13d
  _QWORD *v3; // rax
  _QWORD *v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned int v8; // ebx
  __int64 v10; // rax
  __int64 v11; // rax

  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (_BYTE *)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  v2 = (unsigned __int8)byte_4F96BB0;
  if ( byte_4F96BB0 )
  {
    v2 = 0;
  }
  else
  {
    if ( qword_4F96BA0 || (v10 = sub_131BF10(), (qword_4F96BA0 = (__int64)sub_131C440(a1, v10, 32800, 16)) != 0) )
    {
      if ( qword_4F96BA8 || (v11 = sub_131BF10(), (qword_4F96BA8 = (__int64)sub_131C440(a1, v11, 720, 16)) != 0) )
      {
        v3 = sub_1322110(a1, 4096, 0, 1);
        if ( v3 )
        {
          *((_BYTE *)v3 + 4) = 1;
          v4 = sub_1322110(a1, 4097, 0, 1);
          if ( v4 )
          {
            sub_131DCA0((__int64)v4);
            v5 = qword_4F96BA0;
            *(_DWORD *)(v5 + 8) = sub_1300B70(v4, 4097, v6);
            v7 = qword_4F96BA0;
            if ( !*(_DWORD *)(qword_4F96BA0 + 8) )
            {
LABEL_21:
              *(_QWORD *)(v7 + 16) = 0;
              sub_133A740((__int64)a1);
              byte_4F96BB0 = 1;
              goto LABEL_16;
            }
            v8 = 0;
            while ( sub_1322110(a1, v8, 0, 1) )
            {
              v7 = qword_4F96BA0;
              if ( *(_DWORD *)(qword_4F96BA0 + 8) <= ++v8 )
                goto LABEL_21;
            }
          }
        }
      }
    }
    v2 = 1;
  }
LABEL_16:
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v2;
}
